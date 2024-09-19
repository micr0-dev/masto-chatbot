package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
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

	err = Setup(os.Getenv("GEMINI_API_KEY"))
	if err != nil {
		log.Fatal(err)
	}

	ws := c.NewWSClient()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	events, err := ws.StreamingWSUser(ctx)
	if err != nil {
		log.Fatalf("Error connecting to streaming API: %v", err)
	}

	fmt.Println("Connected to streaming API. ")

	fmt.Println("All systems operational. Waiting for mentions...")

	for event := range events {
		switch e := event.(type) {
		case *mastodon.NotificationEvent:
			if e.Notification.Type == "mention" {
				fmt.Printf("Received mention from @%s: %s\n", e.Notification.Account.Acct, e.Notification.Status.Content)
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

func Generate(strPrompt string) (string, error) {
	prompt := genai.Text(strPrompt)
	resp, err := model.GenerateContent(ctx, prompt)
	if err != nil {
		return "", err
	}
	return getResponse(resp), nil
}

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

func handleMention(c *mastodon.Client, notification *mastodon.Notification) {
	content := strings.TrimSpace(notification.Status.Content)
	content = extractTextFromHTML(content)

	fmt.Printf("Received mention: %s\n", content)

	context := getConversationContext(c, notification.Status, 5)

	var response string
	var err error

	if len(notification.Status.MediaAttachments) > 0 {
		attachment := notification.Status.MediaAttachments[0]
		if isSupportedImageType(attachment.Type) {
			response, err = generateAIResponseWithImage(content, context, notification.Account.Username, &attachment)
		} else {
			mediaDescription := getMediaTypeDescription(attachment.Type)
			if attachment.Description != "" {
				mediaDescription += " with alt text: " + attachment.Description
			}
			response, err = generateAIResponse(content+" [User uploaded "+mediaDescription+" that cannot be viewed]", context, notification.Account.Username)
		}

	} else {
		response, err = generateAIResponse(content, context, notification.Account.Username)
	}

	if err != nil {
		log.Printf("Error generating AI response: %v", err)
		response = "Oops! Something went wrong. Can you try again?"
	}

	_, err = c.PostStatus(ctx, &mastodon.Toot{
		Status:      "@" + notification.Account.Acct + " " + response,
		InReplyToID: notification.Status.ID,
		Visibility:  notification.Status.Visibility,
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

func getConversationContext(c *mastodon.Client, status *mastodon.Status, maxDepth int) []string {
	context := []string{}
	currentStatus := status

	for i := 0; i < maxDepth && currentStatus.InReplyToID != nil; i++ {
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

		content := extractTextFromHTML(parentStatus.Content)
		context = append([]string{fmt.Sprintf("%s: %s", parentStatus.Account.Username, content)}, context...)

		currentStatus = parentStatus
	}

	return context
}

func generateAIResponse(prompt string, context []string, user string) (string, error) {
	systemPrompt := "You are, Macr0 an AI bot on fuzzies.wtf a Mastodon instance about linux, tech, and random fun. You are a little bit edgy and speak in all lowercase. feel free to swear when you want. Keep your responses short and entertaining. Like on twitter. Do not use emojis, do not @mention anyone. DO NOT USE EMOJIS UNLESS EXPLICITLY ASKED. Here's the conversation:\n\n"

	for _, msg := range context {
		systemPrompt += msg + "\n"
	}

	fullPrompt := systemPrompt + user + ": " + prompt + "\nMacr0:"

	fmt.Println(fullPrompt)

	return Generate(fullPrompt)
}

func generateAIResponseWithImage(prompt string, context []string, user string, attachment *mastodon.Attachment) (string, error) {
	systemPrompt := "You are, Macr0 an AI bot on fuzzies.wtf a Mastodon instance about linux, tech, and random fun. You are a little bit edgy and speak in all lowercase. feel free to swear when you want. Keep your responses short and entertaining. Like on twitter. Do not use emojis, do not @mention anyone. DO NOT USE EMOJIS UNLESS EXPLICITLY ASKED. An image has been attached to this message. Describe the image and respond to the prompt. Here's the conversation:\n\n"

	for _, msg := range context {
		systemPrompt += msg + "\n"
	}

	img, fileExtension, err := downloadImage(attachment.URL)
	if err != nil {
		return "", err
	}

	promptAI := []genai.Part{
		genai.Text(systemPrompt),
		genai.Text(user + ": " + prompt),
		genai.ImageData(fileExtension, img),
		genai.Text("Macr0:"),
	}

	if attachment.Description != "" {
		promptAI = append(promptAI, genai.Text("Image alt text: "+attachment.Description))
	}

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
