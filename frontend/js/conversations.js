import {
    apiFetch
} from "./api.js";


let selectedConversationId = null;


export function getSelectedConversationId() {

    return selectedConversationId;

}


export function setSelectedConversationId(
    conversationId
) {

    selectedConversationId =
        conversationId;

}


export async function fetchConversations() {

    const response = await apiFetch(
        "/conversations"
    );

    if (!response.ok) {

        throw new Error(
            "Could not load conversations."
        );

    }

    return response.json();

}


export async function createConversation(
    title = "New Chat"
) {

    const response = await apiFetch(
        "/conversations",
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({
                title
            })
        }
    );


    if (!response.ok) {

        throw new Error(
            "Could not create conversation."
        );

    }


    return response.json();

}


export async function fetchMessages(
    conversationId
) {

    const response = await apiFetch(
        `/conversations/${conversationId}/messages`
    );


    if (!response.ok) {

        throw new Error(
            "Could not load messages."
        );

    }


    const messages =
        await response.json();


    // Backend returns newest first.
    return messages.reverse();

}


export async function deleteConversation(
    conversationId
) {

    const response = await apiFetch(
        `/conversations/${conversationId}`,
        {
            method: "DELETE"
        }
    );


    if (!response.ok) {

        throw new Error(
            "Could not delete conversation."
        );

    }

}