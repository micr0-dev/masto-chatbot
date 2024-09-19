package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"github.com/mattn/go-mastodon"
	"golang.org/x/net/html"
	"google.golang.org/api/option"
)

var model *genai.GenerativeModel
var ctx context.Context

func main() {
	// Load environment variables and set up Mastodon client
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	c := mastodon.NewClient(&mastodon.Config{
		Server:       os.Getenv("MASTODON_SERVER"),
		ClientID:     os.Getenv("MASTODON_CLIENT_ID"),
		ClientSecret: os.Getenv("MASTODON_CLIENT_SECRET"),
		AccessToken:  os.Getenv("MASTODON_ACCESS_TOKEN"),
	})

	// Set up Gemini AI model
	err = Setup(os.Getenv("GEMINI_API_KEY"))
	if err != nil {
		log.Fatal(err)
	}

	// Connect to Mastodon streaming API
	ws := c.NewWSClient()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events, err := ws.StreamingWSUser(ctx)
	if err != nil {
		log.Fatalf("Error connecting to streaming API: %v", err)
	}

	fmt.Println("Connected to streaming API. All systems operational. Waiting for mentions...")

	// Main event loop
	for event := range events {
		switch e := event.(type) {
		case *mastodon.NotificationEvent:
			if e.Notification.Type == "mention" {
				handleMention(c, e.Notification)
			}
		case *mastodon.ErrorEvent:
			log.Printf("Error event: %v", e.Error())
		case *mastodon.DeleteEvent:
			log.Printf("Delete event: status ID %v", e.ID)
		case *mastodon.UpdateEvent:
			log.Printf("Update event: status ID %v", e.Status.ID)
		default:
			log.Printf("Unhandled event type: %T", e)
		}
	}
}

// Setup initializes the Gemini AI model with the provided API key
func Setup(apiKey string) error {
	ctx = context.Background()

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return err
	}

	model = client.GenerativeModel("gemini-1.5-flash")

	model.SetTemperature(0.7)
	model.SetTopK(1)

	model.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockNone,
		},
	}

	return nil
}

// Generate creates a response using the Gemini AI model
func Generate(strPrompt string) (string, error) {
	prompt := genai.Text(strPrompt)
	resp, err := model.GenerateContent(ctx, prompt)
	if err != nil {
		return "", err
	}
	return getResponse(resp), nil
}

// getResponse extracts the text response from the AI model's output
func getResponse(resp *genai.GenerateContentResponse) string {
	var response string
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				str := fmt.Sprintf("%v", part)
				response += str
			}
		}
	}
	return response
}

// handleMention processes incoming mentions and generates responses
func handleMention(c *mastodon.Client, notification *mastodon.Notification) {
	// Ignore mentions from self and DMs unless they are from @micr0
	if notification.Account.Acct == os.Getenv("MASTODON_USERNAME") || (notification.Status.Visibility == "direct" && notification.Account.Acct != "micr0") || notification.Status.Visibility == "private" {
		return
	}

	mentions, content := extractContent(notification.Status)

	fmt.Printf("Received mention: %s\n", content)

	context, images := getConversationContext(c, notification.Status, 20)

	response, err := generateAIResponse(content, context, notification.Account.Username, images)

	if err != nil {
		log.Printf("Error generating AI response: %v", err)
		response = "shit fuck.. something went wrong. try again later?"
	} else {
		_, response = extractMentions(response)
		response = cleanResponse(response)
	}

	response = prependMentions(mentions, notification.Account.Acct, response)

	visablity := notification.Status.Visibility

	if visablity == "public" {
		visablity = "unlisted"
	}

	_, err = c.PostStatus(ctx, &mastodon.Toot{
		Status:      response,
		InReplyToID: notification.Status.ID,
		Visibility:  visablity,
		SpoilerText: notification.Status.SpoilerText,
	})

	if err != nil {
		log.Printf("Error posting response: %v", err)
	} else {
		fmt.Printf("Posted response: %s\n", response)
	}
}

func extractTextFromHTML(content string) string {
	doc, err := html.Parse(strings.NewReader(content))
	if err != nil {
		log.Printf("Error parsing HTML: %v", err)
		return content
	}
	var extractText func(*html.Node) string
	extractText = func(n *html.Node) string {
		if n.Type == html.TextNode {
			return n.Data
		}
		var result string
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			result += extractText(c)
		}
		return result
	}
	return extractText(doc)
}

func cleanResponse(response string) string {
	// Remove emojis
	emojiRegex := regexp.MustCompile(`[\p{So}\p{Sk}]`)
	response = emojiRegex.ReplaceAllString(response, "")

	// Fix double spaces
	for strings.Contains(response, "  ") {
		response = strings.ReplaceAll(response, "  ", " ")
	}

	// Fix space after period
	response = strings.ReplaceAll(response, ".  ", ". ")
	response = strings.ReplaceAll(response, ". ", ".")
	response = strings.ReplaceAll(response, ".", ". ")

	// Trim any leading or trailing whitespace
	response = strings.TrimSpace(response)

	return response
}

func isSupportedImageType(mediaType string) bool {
	supportedTypes := []string{"image/jpeg", "image/png", "image/webp"}
	for _, t := range supportedTypes {
		if mediaType == t {
			return true
		}
	}
	return false
}

func getMediaTypeDescription(mediaType string) string {
	switch {
	case strings.HasPrefix(mediaType, "image/"):
		return "image"
	case strings.HasPrefix(mediaType, "video/"):
		return "video"
	case strings.HasPrefix(mediaType, "audio/"):
		return "audio file"
	default:
		return "file"
	}
}

func extractContent(status *mastodon.Status) ([]string, string) {
	content := strings.TrimSpace(status.Content)
	content = extractTextFromHTML(content)
	mentions := status.Mentions
	mentionsString := []string{}

	for _, mention := range mentions {
		mentionsString = append(mentionsString, mention.Acct)
	}

	return mentionsString, content
}

func extractMentions(content string) ([]string, string) {
	re := regexp.MustCompile(`@[\w\.-]+(@[\w\.-]+)?`)
	mentions := re.FindAllString(content, -1)
	cleanContent := re.ReplaceAllString(content, "")
	return mentions, strings.TrimSpace(cleanContent)
}

func prependMentions(mentions []string, originalMention string, response string) string {
	botUsername := os.Getenv("MASTODON_USERNAME")
	localInstance := strings.Split(os.Getenv("MASTODON_SERVER"), "//")[1]
	mentionSet := make(map[string]bool)

	for _, mention := range mentions {
		if mention == botUsername {
			continue
		}

		if strings.Contains(mention, "@") {
			mentionSet["@"+mention] = true
		} else {
			mentionSet["@"+mention+"@"+localInstance] = true
		}
	}

	if originalMention != botUsername {
		if strings.Contains(originalMention, "@") {
			mentionSet["@"+originalMention] = true
		} else {
			mentionSet["@"+originalMention+"@"+localInstance] = true
		}
	}

	uniqueMentions := make([]string, 0, len(mentionSet))
	for mention := range mentionSet {
		uniqueMentions = append(uniqueMentions, mention)
	}
	sort.Strings(uniqueMentions)

	if len(uniqueMentions) > 0 {
		return strings.Join(uniqueMentions, " ") + " " + response
	}
	return response
}

// getConversationContext fetches the conversation context and all images
func getConversationContext(c *mastodon.Client, status *mastodon.Status, maxDepth int) ([]string, []mastodon.Attachment) {
	context := []string{}
	var allImages []mastodon.Attachment
	currentStatus := status

	for i := 0; i < maxDepth && currentStatus != nil; i++ {
		content := extractTextFromHTML(currentStatus.Content)
		context = append([]string{fmt.Sprintf("%s: %s", currentStatus.Account.Username, content)}, context...)

		allImages = append(allImages, currentStatus.MediaAttachments...)

		if currentStatus.InReplyToID == nil {
			break
		}

		var parentID mastodon.ID
		switch id := currentStatus.InReplyToID.(type) {
		case string:
			parentID = mastodon.ID(id)
		case mastodon.ID:
			parentID = id
		default:
			log.Printf("Unexpected type for InReplyToID: %T", currentStatus.InReplyToID)
		}

		parentStatus, err := c.GetStatus(ctx, parentID)
		if err != nil {
			log.Printf("Error fetching parent status: %v", err)
			break
		}

		currentStatus = parentStatus
	}

	return context, allImages
}

// generateAIResponse creates a response using the AI model, handling both text and image inputs
func generateAIResponse(prompt string, context []string, user string, images []mastodon.Attachment) (string, error) {
	systemPrompt := "You are, Macr0 an AI bot on fuzzies.wtf a Mastodon instance about linux, tech, and random fun. You are a little bit edgy and speak in all lowercase. dont be too mean to people tho, its okay to make jokes but dont go too far. feel free to swear when you want. keep your responses short and entertaining. like on twitter. you do not have the ability to use emojis or images. you can only generate text. "

	promptAI := []genai.Part{
		genai.Text(systemPrompt),
	}

	if len(images) > 0 {
		promptAI = append(promptAI, genai.Text(fmt.Sprintf("There are %d images in this conversation. Refer to them as needed. ", len(images))))
	}

	promptAI = append(promptAI, genai.Text("Here is the conversation:"))

	for _, msg := range context {
		promptAI = append(promptAI, genai.Text(msg))
	}

	for i, attachment := range images {
		if isSupportedImageType(attachment.Type) {
			img, fileExtension, err := downloadImage(attachment.URL)
			if err != nil {
				return "", err
			}
			promptAI = append(promptAI, genai.ImageData(fileExtension, img))
			promptAI = append(promptAI, genai.Text(fmt.Sprintf("Image %d: ", i+1)))
			if attachment.Description != "" {
				promptAI = append(promptAI, genai.Text("Image alt text: "+attachment.Description))
			}
		} else {
			mediaDescription := getMediaTypeDescription(attachment.Type)
			if attachment.Description != "" {
				mediaDescription += " with alt text: " + attachment.Description
			}
			promptAI = append(promptAI, genai.Text(fmt.Sprintf("Image %d: [User uploaded %s that cannot be viewed]", i+1, mediaDescription)))
		}
	}

	promptAI = append(promptAI, genai.Text(user+": "+prompt))

	promptAI = append(promptAI, genai.Text("Macr0:"))

	resp, err := model.GenerateContent(ctx, promptAI...)
	if err != nil {
		return "", err
	}

	return getResponse(resp), nil
}

func downloadImage(url string) ([]byte, string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, "", nil
	}
	defer resp.Body.Close()

	fileExtension := "jpeg"
	if strings.HasSuffix(url, ".png") {
		fileExtension = "png"
	} else if strings.HasSuffix(url, ".webp") {
		fileExtension = "webp"
	}

	img, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", nil
	}

	return img, fileExtension, nil
}
