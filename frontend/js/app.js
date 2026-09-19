import {
    apiFetch,
    getAccessToken
} from "./api.js";

import {
    initializeAuth
} from "./auth.js";

import {
    createConversation,
    deleteConversation,
    fetchConversations,
    fetchMessages,
    getSelectedConversationId,
    setSelectedConversationId
} from "./conversations.js";

import {
    streamChat
} from "./chat.js";


/* ================================================================
   Application state
================================================================ */

let selectedResumeId = null;
let isGenerating = false;


/* ================================================================
   DOM elements
================================================================ */

const authScreen =
    document.getElementById("auth-screen");

const app =
    document.getElementById("app");

const conversationList =
    document.getElementById("conversation-list");

const conversationTitle =
    document.getElementById("conversation-title");

const messagesContainer =
    document.getElementById("messages");

const welcomeMessage =
    document.getElementById("welcome-message");

const chatForm =
    document.getElementById("chat-form");

const messageInput =
    document.getElementById("message-input");

const sendButton =
    document.getElementById("send-button");

const newChatButton =
    document.getElementById("new-chat-button");

const agentStatus =
    document.getElementById("agent-status");

const resumeButton =
    document.getElementById("resume-button");

const resumeModal =
    document.getElementById("resume-modal");

const closeResumeModalButton =
    document.getElementById("close-resume-modal");

const resumeUploadForm =
    document.getElementById("resume-upload-form");

const resumeFileInput =
    document.getElementById("resume-file");

const resumeList =
    document.getElementById("resume-list");

const selectedResumeLabel =
    document.getElementById("selected-resume-label");


/* ================================================================
   Authentication UI
================================================================ */

function showApplication() {

    authScreen.classList.add("hidden");

    app.classList.remove("hidden");

}


function showAuthentication() {

    app.classList.add("hidden");

    authScreen.classList.remove("hidden");

    setSelectedConversationId(null);

    selectedResumeId = null;

    clearMessages();

}


/* ================================================================
   Messages UI
================================================================ */

function clearMessages() {

    messagesContainer.innerHTML = "";

}


function showWelcomeMessage() {

    clearMessages();

    messagesContainer.appendChild(
        createWelcomeElement()
    );

}


function createWelcomeElement() {

    const container =
        document.createElement("div");

    container.className = "welcome";

    container.innerHTML = `
        <h2>How can CareerAI help?</h2>

        <p>
            Ask about careers, resumes, skills,
            jobs, or interview preparation.
        </p>
    `;

    return container;

}


function createMessageElement(
    role,
    content
) {

    const wrapper =
        document.createElement("div");

    wrapper.classList.add(
        "message-row"
    );

    if (role === "user") {

        wrapper.classList.add(
            "user-message-row"
        );

    }
    else {

        wrapper.classList.add(
            "assistant-message-row"
        );

    }


    const message =
        document.createElement("div");

    message.classList.add(
        "message"
    );

    message.classList.add(
        role === "user"
            ? "user-message"
            : "assistant-message"
    );


    /*
        Using textContent rather than innerHTML prevents
        arbitrary HTML returned by the model from executing.
    */
    if (role == "assistant") {

        renderAssistantMarkdown(
            message,
            content || ""
        );
        
    }
    else {

        message.textContent = content || "";

    }


    wrapper.appendChild(
        message
    );


    messagesContainer.appendChild(
        wrapper
    );


    scrollToBottom();


    return message;

}


function scrollToBottom() {

    messagesContainer.scrollTop =
        messagesContainer.scrollHeight;

}


function renderAssistantMarkdown(
    element,
    markdown
) {

    const html =
        marked.parse(markdown);

    element.innerHTML =
        DOMPurify.sanitize(html);


    /*
        Open links in a new tab.
    */
    const links =
        element.querySelectorAll("a");

    for (const link of links) {

        link.target =
            "_blank";

        link.rel =
            "noopener noreferrer";

    }

}

/* ================================================================
   Agent status
================================================================ */

function showAgentStatus(
    message
) {

    agentStatus.textContent =
        message;

    agentStatus.classList.remove(
        "hidden"
    );

}


function hideAgentStatus() {

    agentStatus.classList.add(
        "hidden"
    );

    agentStatus.textContent = "";

}


/* ================================================================
   Conversation sidebar
================================================================ */

async function loadConversations(
    preferredConversationId = null
) {

    const conversations =
        await fetchConversations();


    renderConversationList(
        conversations
    );


    if (
        preferredConversationId !== null
    ) {

        const conversation =
            conversations.find(
                item =>
                    item.id ===
                    preferredConversationId
            );

        if (conversation) {

            await selectConversation(
                conversation
            );

            return;

        }

    }


    const currentId =
        getSelectedConversationId();


    if (currentId !== null) {

        const existing =
            conversations.find(
                item =>
                    item.id === currentId
            );

        if (existing) {

            highlightSelectedConversation();

            return;

        }

    }


    /*
        We intentionally do not automatically open
        the newest conversation.

        This preserves the ChatGPT-style welcome
        state until the user chooses a conversation
        or creates a new one.
    */

    showWelcomeMessage();

}


function renderConversationList(
    conversations
) {

    conversationList.innerHTML = "";


    for (
        const conversation
        of conversations
    ) {

        const item =
            document.createElement("div");

        item.className =
            "conversation-item";

        item.dataset.conversationId =
            conversation.id;


        const title =
            document.createElement("button");

        title.className =
            "conversation-title-button";

        title.textContent =
            conversation.title
            || "New Chat";


        title.addEventListener(
            "click",
            async () => {

                await selectConversation(
                    conversation
                );

            }
        );


        const deleteButton =
            document.createElement("button");

        deleteButton.className =
            "conversation-delete-button";

        deleteButton.textContent =
            "×";

        deleteButton.title =
            "Delete conversation";


        deleteButton.addEventListener(
            "click",
            async event => {

                event.stopPropagation();

                await handleDeleteConversation(
                    conversation.id
                );

            }
        );


        item.append(
            title,
            deleteButton
        );


        conversationList.appendChild(
            item
        );

    }


    highlightSelectedConversation();

}


function highlightSelectedConversation() {

    const selectedId =
        getSelectedConversationId();


    const items =
        conversationList.querySelectorAll(
            ".conversation-item"
        );


    for (const item of items) {

        const itemId =
            Number(
                item.dataset.conversationId
            );


        item.classList.toggle(
            "active",
            itemId === selectedId
        );

    }

}


/* ================================================================
   Select conversation
================================================================ */

async function selectConversation(
    conversation
) {

    if (isGenerating) {
        return;
    }


    setSelectedConversationId(
        conversation.id
    );


    conversationTitle.textContent = "CareerAI"

    highlightSelectedConversation();


    await loadConversationResume(
        conversation.id
    );


    await loadConversationMessages(
        conversation.id
    );

}


async function loadConversationResume(
    conversationId
) {

    try {

        const resume =
            await fetchConversationResume(
                conversationId
            );


        if (
            resume.resume_id === null
        ) {

            selectedResumeId = null;

            updateSelectedResumeLabel();

            return;

        }


        selectedResumeId =
            resume.resume_id;


        selectedResumeLabel.textContent =
            `Resume: ${resume.filename}`;


        selectedResumeLabel.classList.add(
            "active"
        );

    }
    catch (error) {

        selectedResumeId = null;

        updateSelectedResumeLabel();

        console.error(error);

    }

}

/* ================================================================
   Load persisted message history
================================================================ */

async function loadConversationMessages(
    conversationId
) {

    clearMessages();

    showAgentStatus(
        "Loading conversation..."
    );


    try {

        const messages =
            await fetchMessages(
                conversationId
            );


        hideAgentStatus();


        if (messages.length === 0) {

            showWelcomeMessage();

            return;

        }


        for (const message of messages) {

            if (
                message.role !== "user"
                && message.role !== "assistant"
            ) {

                continue;

            }


            createMessageElement(
                message.role,
                message.content
            );

        }

    }
    catch (error) {

        hideAgentStatus();

        createMessageElement(
            "assistant",
            "Could not load this conversation."
        );

        console.error(error);

    }

}


/* ================================================================
   Create conversation
================================================================ */

function handleNewConversation() {

    if (isGenerating) {
        return;
    }


    /*
        Do not create a database conversation yet.

        A conversation will be created only when
        the user sends the first message.
    */
    setSelectedConversationId(
        null
    );


    conversationTitle.textContent =
        "New Chat";


    /*
        A new conversation starts without
        a selected resume.
    */
    selectedResumeId = null;

    updateSelectedResumeLabel();


    /*
        Remove the active highlight from the
        previously selected conversation.
    */
    highlightSelectedConversation();


    /*
        Show the empty welcome screen.
    */
    showWelcomeMessage();


    messageInput.value = "";

    autoResizeTextarea();

    messageInput.focus();

}


/* ================================================================
   Delete conversation
================================================================ */

async function handleDeleteConversation(
    conversationId
) {

    if (isGenerating) {
        return;
    }


    const confirmed =
        window.confirm(
            "Delete this conversation?"
        );


    if (!confirmed) {
        return;
    }


    try {

        await deleteConversation(
            conversationId
        );


        if (
            getSelectedConversationId()
            === conversationId
        ) {

            setSelectedConversationId(
                null
            );

            conversationTitle.textContent =
                "CareerAI";

            showWelcomeMessage();

        }


        await loadConversations();

    }
    catch (error) {

        console.error(error);

    }

}


/* ================================================================
   Chat streaming
================================================================ */

async function handleSendMessage(
    event
) {

    event.preventDefault();


    if (isGenerating) {
        return;
    }


    const userMessage =
        messageInput.value.trim();


    if (!userMessage) {
        return;
    }


    let conversationId =
        getSelectedConversationId();


    /*
        If the user starts typing directly from the
        welcome screen, automatically create a chat.
    */
    if (conversationId === null) {

        try {

            const conversation =
                await createConversation(
                    createConversationTitle(
                        userMessage
                    )
                );


            conversationId =
                conversation.id;


            setSelectedConversationId(
                conversationId
            );


            conversationTitle.textContent =
                conversation.title
                || "New Chat";


            await loadConversations(
                conversationId
            );

        }
        catch (error) {

            console.error(error);

            return;

        }

    }


    removeWelcomeIfPresent();


    createMessageElement(
        "user",
        userMessage
    );


    messageInput.value = "";

    autoResizeTextarea();


    /*
        Create an empty assistant bubble.
        Streaming tokens will be appended to this.
    */
    const assistantElement =
        createMessageElement(
            "assistant",
            ""
        );


    isGenerating = true;

    sendButton.disabled = true;
    messageInput.disabled = true;


    let receivedResponseToken = false;

    let assistantMarkdown = "";


    try {

        await streamChat({

            conversationId,

            message: userMessage,

            resumeId:
                selectedResumeId,

            onStatus: message => {

                showAgentStatus(
                    message
                );

            },

            onToken: token => {

                if (
                    !receivedResponseToken
                ) {

                    receivedResponseToken =
                        true;

                    hideAgentStatus();

                }


                assistantMarkdown += token;

                renderAssistantMarkdown(
                    assistantElement,
                    assistantMarkdown
                );


                scrollToBottom();

            },

            onJobResults: results => {

                /*
                    job_results are already stored by
                    the backend.

                    Later we can render rich job cards
                    here if desired.
                */

                console.log(
                    "Structured job results:",
                    results
                );

            },

            onDone: () => {

                hideAgentStatus();

            },

            onError: message => {

                hideAgentStatus();

                if (
                    !assistantElement.textContent
                ) {

                    assistantElement.textContent =
                        message;

                }

            }

        });


        /*
            Refresh sidebar in case we later implement
            automatic conversation-title updates.
        */
        await loadConversations(
            conversationId
        );

    }
    catch (error) {

        hideAgentStatus();


        if (
            !assistantElement.textContent
        ) {

            assistantElement.textContent =
                error.message
                || "CareerAI could not complete the request.";

        }


        console.error(error);

    }
    finally {

        isGenerating = false;

        sendButton.disabled = false;
        messageInput.disabled = false;

        messageInput.focus();

    }

}


function removeWelcomeIfPresent() {

    const welcome =
        messagesContainer.querySelector(
            ".welcome"
        );

    if (welcome) {
        welcome.remove();
    }

}


function createConversationTitle(
    message
) {

    const maxLength = 45;


    if (
        message.length <= maxLength
    ) {

        return message;

    }


    return (
        message.slice(
            0,
            maxLength
        ).trim()
        + "..."
    );

}


/* ================================================================
   Resume API
================================================================ */

async function fetchResumes() {

    const response =
        await apiFetch(
            "/resumes"
        );


    if (!response.ok) {

        throw new Error(
            "Could not load resumes."
        );

    }


    return response.json();

}


async function uploadResume(
    file
) {

    const formData =
        new FormData();


    formData.append(
        "file",
        file
    );


    const response =
        await apiFetch(
            "/resumes",
            {
                method: "POST",
                body: formData
            }
        );


    if (!response.ok) {

        let message =
            "Resume upload failed.";


        try {

            const error =
                await response.json();

            message =
                error.detail
                || message;

        }
        catch {
            // Ignore malformed response body.
        }


        throw new Error(
            message
        );

    }


    return response.json();

}


async function deleteResume(
    resumeId
) {

    const response =
        await apiFetch(
            `/resumes/${resumeId}`,
            {
                method: "DELETE"
            }
        );


    if (!response.ok) {

        throw new Error(
            "Could not delete resume."
        );

    }

}


/* ================================================================
   Resume modal
================================================================ */

async function openResumeModal() {

    resumeModal.classList.remove(
        "hidden"
    );


    await loadResumes();

}


function closeResumeModal() {

    resumeModal.classList.add(
        "hidden"
    );

}


async function loadResumes() {

    resumeList.innerHTML =
        "<p>Loading resumes...</p>";


    try {

        const resumes =
            await fetchResumes();


        renderResumes(
            resumes
        );

    }
    catch (error) {

        resumeList.innerHTML =
            "<p>Could not load resumes.</p>";

        console.error(error);

    }

}


function renderResumes(
    resumes
) {

    resumeList.innerHTML = "";


    if (
        resumes.length === 0
    ) {

        const empty =
            document.createElement("p");

        empty.textContent =
            "No resumes uploaded yet.";

        resumeList.appendChild(
            empty
        );

        return;

    }


    for (const resume of resumes) {

        const item =
            document.createElement("div");

        item.className =
            "resume-item";


        if (
            resume.id ===
            selectedResumeId
        ) {

            item.classList.add(
                "selected"
            );

        }


        const info =
            document.createElement("button");

        info.className =
            "resume-select-button";

        info.textContent =
            resume.filename
            || `Resume ${resume.id}`;


        info.addEventListener(
            "click",
            () => {

                selectResume(
                    resume
                );

                closeResumeModal();

            }
        );


        const deleteButton =
            document.createElement("button");

        deleteButton.className =
            "resume-delete-button";

        deleteButton.textContent =
            "Delete";


        deleteButton.addEventListener(
            "click",
            async event => {

                event.stopPropagation();


                const confirmed =
                    window.confirm(
                        `Delete ${resume.filename}?`
                    );


                if (!confirmed) {
                    return;
                }


                try {

                    await deleteResume(
                        resume.id
                    );


                    if (
                        selectedResumeId
                        === resume.id
                    ) {

                        selectedResumeId =
                            null;

                        updateSelectedResumeLabel();

                    }


                    await loadResumes();

                }
                catch (error) {

                    console.error(error);

                }

            }
        );


        item.append(
            info,
            deleteButton
        );


        resumeList.appendChild(
            item
        );

    }

}


function selectResume(
    resume
) {

    selectedResumeId =
        resume.id;


    selectedResumeLabel.textContent =
        `Resume: ${resume.filename}`;


    selectedResumeLabel.classList.add(
        "active"
    );

}


function updateSelectedResumeLabel() {

    if (
        selectedResumeId === null
    ) {

        selectedResumeLabel.textContent =
            "No resume selected";

        selectedResumeLabel.classList.remove(
            "active"
        );

    }

}


/* ================================================================
   Resume upload
================================================================ */

async function handleResumeUpload(
    event
) {

    event.preventDefault();


    const file =
        resumeFileInput.files[0];


    if (!file) {
        return;
    }


    try {

        resumeList.innerHTML =
            "<p>Processing resume...</p>";


        /*
            This request may take longer than an ordinary upload
            because the backend also creates:
            - FAISS resume embeddings
            - GraphRAG entities/relationships
        */
        const resume =
            await uploadResume(
                file
            );


        selectResume(
            resume
        );


        resumeUploadForm.reset();


        await loadResumes();

    }
    catch (error) {

        resumeList.innerHTML =
            `<p>${error.message}</p>`;

        console.error(error);

    }

}


/* ================================================================
   Textarea
================================================================ */

function autoResizeTextarea() {

    messageInput.style.height =
        "auto";


    messageInput.style.height =
        `${Math.min(
            messageInput.scrollHeight,
            180
        )}px`;

}


/* ================================================================
   Keyboard behavior
================================================================ */

function initializeComposer() {

    messageInput.addEventListener(
        "input",
        autoResizeTextarea
    );


    messageInput.addEventListener(
        "keydown",
        event => {

            /*
                Enter      => send
                Shift+Enter => newline
            */
            if (
                event.key === "Enter"
                && !event.shiftKey
            ) {

                event.preventDefault();

                chatForm.requestSubmit();

            }

        }
    );


    chatForm.addEventListener(
        "submit",
        handleSendMessage
    );

}


/* ================================================================
   Event listeners
================================================================ */

function initializeEventListeners() {

    newChatButton.addEventListener(
        "click",
        handleNewConversation
    );


    resumeButton.addEventListener(
        "click",
        openResumeModal
    );


    closeResumeModalButton.addEventListener(
        "click",
        closeResumeModal
    );


    resumeUploadForm.addEventListener(
        "submit",
        handleResumeUpload
    );


    /*
        Clicking outside the modal closes it.
    */
    resumeModal.addEventListener(
        "click",
        event => {

            if (
                event.target ===
                resumeModal
            ) {

                closeResumeModal();

            }

        }
    );

}


/* ================================================================
   Application startup
================================================================ */

async function initializeApplication() {

    try {

        showApplication();

        await loadConversations();

        updateSelectedResumeLabel();

    }
    catch (error) {

        console.error(error);

        showAuthentication();

    }

}


async function fetchConversationResume(
    conversationId
) {

    const response =
        await apiFetch(
            `/chat/conversations/${conversationId}/resume`
        );


    if (!response.ok) {

        throw new Error(
            "Could not load selected resume."
        );

    }


    return response.json();

}


/* ================================================================
   Initialize auth
================================================================ */

initializeAuth({

    onAuthenticated:
        async () => {

            await initializeApplication();

        },

    onLoggedOut:
        () => {

            showAuthentication();

        }

});


initializeComposer();

initializeEventListeners();


/*
    If an access token already exists, try opening
    the application immediately.

    apiFetch will attempt refresh automatically when
    the access token has expired.
*/
if (getAccessToken()) {

    initializeApplication();

}
else {

    showAuthentication();

}