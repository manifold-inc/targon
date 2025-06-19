package discord

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/centrifuge/go-substrate-rpc-client/v4/types"
)

func LogWeightsToDiscord(disc_url string, uids, scores []types.U16, h types.Header) error {
	color := "3447003"
	title := fmt.Sprintf("Validator setting weights at block %v", h.Number)
	desc := fmt.Sprintf("UIDS: %v\n\nweights: %v", uids, scores)
	uname := "Validator Logs"
	msg := Message{
		Username: &uname,
		Embeds: &[]Embed{{
			Title:       &title,
			Description: &desc,
			Color:       &color,
		}},
	}
	err := SendDiscordMessage(disc_url, msg)
	if err != nil {
		return err
	}
	return nil
}

func SendDiscordMessage(url string, message Message) error {
	if len(url) == 0 {
		return nil
	}
	payload := new(bytes.Buffer)

	err := json.NewEncoder(payload).Encode(message)
	if err != nil {
		return err
	}

	resp, err := http.Post(url, "application/json", payload)
	if err != nil {
		return err
	}

	if resp.StatusCode != 200 && resp.StatusCode != 204 {
		defer func() {
			_ = resp.Body.Close()
		}()

		responseBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}

		return fmt.Errorf("%s", responseBody)
	}

	return nil
}

type Message struct {
	Username        *string          `json:"username,omitempty"`
	AvatarUrl       *string          `json:"avatar_url,omitempty"`
	Content         *string          `json:"content,omitempty"`
	Embeds          *[]Embed         `json:"embeds,omitempty"`
	AllowedMentions *AllowedMentions `json:"allowed_mentions,omitempty"`
}

type Embed struct {
	Title       *string    `json:"title,omitempty"`
	Url         *string    `json:"url,omitempty"`
	Description *string    `json:"description,omitempty"`
	Color       *string    `json:"color,omitempty"`
	Author      *Author    `json:"author,omitempty"`
	Fields      *[]Field   `json:"fields,omitempty"`
	Thumbnail   *Thumbnail `json:"thumbnail,omitempty"`
	Image       *Image     `json:"image,omitempty"`
	Footer      *Footer    `json:"footer,omitempty"`
}

type Author struct {
	Name    *string `json:"name,omitempty"`
	Url     *string `json:"url,omitempty"`
	IconUrl *string `json:"icon_url,omitempty"`
}

type Field struct {
	Name   *string `json:"name,omitempty"`
	Value  *string `json:"value,omitempty"`
	Inline *bool   `json:"inline,omitempty"`
}

type Thumbnail struct {
	Url *string `json:"url,omitempty"`
}

type Image struct {
	Url *string `json:"url,omitempty"`
}

type Footer struct {
	Text    *string `json:"text,omitempty"`
	IconUrl *string `json:"icon_url,omitempty"`
}

type AllowedMentions struct {
	Parse *[]string `json:"parse,omitempty"`
	Users *[]string `json:"users,omitempty"`
	Roles *[]string `json:"roles,omitempty"`
}
