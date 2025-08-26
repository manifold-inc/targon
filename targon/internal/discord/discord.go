// Package discord
package discord

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

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
	AvatarURL       *string          `json:"avatar_url,omitempty"`
	Content         *string          `json:"content,omitempty"`
	Embeds          *[]Embed         `json:"embeds,omitempty"`
	AllowedMentions *AllowedMentions `json:"allowed_mentions,omitempty"`
}

type Embed struct {
	Title       *string    `json:"title,omitempty"`
	URL         *string    `json:"url,omitempty"`
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
	URL     *string `json:"url,omitempty"`
	IconURL *string `json:"icon_url,omitempty"`
}

type Field struct {
	Name   *string `json:"name,omitempty"`
	Value  *string `json:"value,omitempty"`
	Inline *bool   `json:"inline,omitempty"`
}

type Thumbnail struct {
	URL *string `json:"url,omitempty"`
}

type Image struct {
	URL *string `json:"url,omitempty"`
}

type Footer struct {
	Text    *string `json:"text,omitempty"`
	IconURL *string `json:"icon_url,omitempty"`
}

type AllowedMentions struct {
	Parse *[]string `json:"parse,omitempty"`
	Users *[]string `json:"users,omitempty"`
	Roles *[]string `json:"roles,omitempty"`
}
