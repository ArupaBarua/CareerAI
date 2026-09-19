import {
    apiFetch
} from "./api.js";


export async function streamChat({
    conversationId,
    message,
    resumeId,
    onStatus,
    onToken,
    onJobResults,
    onDone,
    onError
}) {

    const payload = {
        message
    };


    if (resumeId !== null) {

        payload.resume_id =
            resumeId;

    }


    const response = await apiFetch(
        `/chat/conversations/${conversationId}/stream`,
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body:
                JSON.stringify(
                    payload
                )
        }
    );


    if (!response.ok) {

        let message =
            "CareerAI request failed.";

        try {

            const error =
                await response.json();

            message =
                error.detail || message;

        }

        catch {
            // Ignore malformed error bodies.
        }


        throw new Error(message);

    }


    if (!response.body) {

        throw new Error(
            "Streaming is not supported."
        );

    }


    const reader =
        response.body.getReader();

    const decoder =
        new TextDecoder();


    let buffer = "";


    while (true) {

        const {
            value,
            done
        } = await reader.read();


        if (done) {
            break;
        }


        buffer += decoder.decode(
            value,
            {
                stream: true
            }
        );


        // SSE events end with a blank line.
        const blocks =
            buffer.split("\n\n");


        // Keep incomplete event.
        buffer = blocks.pop() || "";


        for (
            const block
            of blocks
        ) {

            processSSEBlock(
                block,
                {
                    onStatus,
                    onToken,
                    onJobResults,
                    onDone,
                    onError
                }
            );

        }

    }


    if (buffer.trim()) {

        processSSEBlock(
            buffer,
            {
                onStatus,
                onToken,
                onJobResults,
                onDone,
                onError
            }
        );

    }

}


function processSSEBlock(
    block,
    callbacks
) {

    const lines =
        block.split("\n");


    let eventType =
        "message";

    const dataLines = [];


    for (const line of lines) {

        if (
            line.startsWith(
                "event:"
            )
        ) {

            eventType =
                line
                    .slice(6)
                    .trim();

        }

        else if (
            line.startsWith(
                "data:"
            )
        ) {

            dataLines.push(
                line
                    .slice(5)
                    .trimStart()
            );

        }

    }


    if (
        dataLines.length === 0
    ) {

        return;

    }


    let data;


    try {

        data = JSON.parse(
            dataLines.join("\n")
        );

    }

    catch {

        return;

    }


    switch (eventType) {

        case "status":

            callbacks.onStatus?.(
                data.message
            );

            break;


        case "response_start":

            break;


        case "token":

            callbacks.onToken?.(
                data.content
            );

            break;


        case "job_results":

            callbacks.onJobResults?.(
                data.results
            );

            break;


        case "done":

            callbacks.onDone?.(
                data
            );

            break;


        case "error":

            callbacks.onError?.(
                data.message
            );

            break;

    }

}