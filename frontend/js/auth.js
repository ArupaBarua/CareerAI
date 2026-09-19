import {
    API_BASE_URL,
    clearAccessToken,
    setAccessToken
} from "./api.js";


let authMode = "login";


/* ================================================================
   Error handling
================================================================ */

function getErrorMessage(
    errorData,
    fallbackMessage
) {

    if (!errorData) {
        return fallbackMessage;
    }


    /*
        Normal FastAPI errors:

        {
            "detail": "Username already exists."
        }
    */
    if (
        typeof errorData.detail === "string"
    ) {

        return errorData.detail;

    }


    /*
        Pydantic / FastAPI validation errors:

        {
            "detail": [
                {
                    "loc": ["body", "password"],
                    "msg": "String should have at least 8 characters",
                    ...
                }
            ]
        }
    */
    if (
        Array.isArray(
            errorData.detail
        )
    ) {

        return errorData.detail
            .map(error => {

                const field =
                    error.loc?.[
                        error.loc.length - 1
                    ];


                if (field) {

                    return (
                        `${formatFieldName(field)}: `
                        + error.msg
                    );

                }


                return error.msg;

            })
            .join("\n");

    }


    return fallbackMessage;

}


function formatFieldName(
    field
) {

    return field
        .replaceAll("_", " ")
        .replace(
            /\b\w/g,
            character =>
                character.toUpperCase()
        );

}


/* ================================================================
   Initialization
================================================================ */

export function initializeAuth({
    onAuthenticated,
    onLoggedOut
}) {

    const loginTab =
        document.getElementById(
            "login-tab"
        );

    const registerTab =
        document.getElementById(
            "register-tab"
        );

    const authForm =
        document.getElementById(
            "auth-form"
        );

    const authSubmit =
        document.getElementById(
            "auth-submit"
        );

    const usernameInput =
        document.getElementById(
            "username"
        );

    const passwordInput =
        document.getElementById(
            "password"
        );

    const errorElement =
        document.getElementById(
            "auth-error"
        );


    /* ============================================================
       Login tab
    ============================================================ */

    loginTab.addEventListener(
        "click",
        () => {

            authMode = "login";


            loginTab.classList.add(
                "active"
            );

            registerTab.classList.remove(
                "active"
            );


            authSubmit.textContent =
                "Login";


            /*
                Clear anything typed while on
                the Register form.
            */
            authForm.reset();


            errorElement.textContent =
                "";


            usernameInput.placeholder =
                "Username";

            passwordInput.placeholder =
                "Password";


            usernameInput.autocomplete =
                "username";

            passwordInput.autocomplete =
                "current-password";


            usernameInput.focus();

        }
    );


    /* ============================================================
       Register tab
    ============================================================ */

    registerTab.addEventListener(
        "click",
        () => {

            authMode = "register";


            registerTab.classList.add(
                "active"
            );

            loginTab.classList.remove(
                "active"
            );


            authSubmit.textContent =
                "Register";


            /*
                Clear anything typed while on
                the Login form.
            */
            authForm.reset();


            errorElement.textContent =
                "";


            usernameInput.placeholder =
                "Choose a username";

            passwordInput.placeholder =
                "Create a password";


            usernameInput.autocomplete =
                "username";

            passwordInput.autocomplete =
                "new-password";


            usernameInput.focus();

        }
    );


    /* ============================================================
       Submit Login / Register
    ============================================================ */

    authForm.addEventListener(
        "submit",
        async event => {

            event.preventDefault();


            errorElement.textContent =
                "";


            const username =
                usernameInput.value.trim();

            const password =
                passwordInput.value;


            authSubmit.disabled = true;


            try {

                /* ------------------------------------------------
                   Register first
                ------------------------------------------------ */

                if (
                    authMode ===
                    "register"
                ) {

                    const registerResponse =
                        await fetch(
                            `${API_BASE_URL}/auth/register`,
                            {
                                method: "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                body:
                                    JSON.stringify({
                                        username,
                                        password
                                    })
                            }
                        );


                    if (
                        !registerResponse.ok
                    ) {

                        let errorData = null;


                        try {

                            errorData =
                                await registerResponse.json();

                        }
                        catch {
                            // Response was not JSON.
                        }


                        throw new Error(
                            getErrorMessage(
                                errorData,
                                "Registration failed."
                            )
                        );

                    }

                }


                /* ------------------------------------------------
                   Login
                ------------------------------------------------ */

                const loginResponse =
                    await fetch(
                        `${API_BASE_URL}/auth/login`,
                        {
                            method: "POST",

                            credentials:
                                "include",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify({
                                    username,
                                    password
                                })
                        }
                    );


                if (
                    !loginResponse.ok
                ) {

                    let errorData = null;


                    try {

                        errorData =
                            await loginResponse.json();

                    }
                    catch {
                        // Response was not JSON.
                    }


                    throw new Error(
                        getErrorMessage(
                            errorData,
                            "Login failed."
                        )
                    );

                }


                const data =
                    await loginResponse.json();


                setAccessToken(
                    data.access_token
                );


                authForm.reset();


                await onAuthenticated();

            }
            catch (error) {

                errorElement.textContent =
                    error.message;

            }
            finally {

                authSubmit.disabled =
                    false;

            }

        }
    );


    /* ============================================================
       Logout
    ============================================================ */

    document
        .getElementById(
            "logout-button"
        )
        .addEventListener(
            "click",
            async () => {

                try {

                    await fetch(
                        `${API_BASE_URL}/auth/logout`,
                        {
                            method: "POST",

                            credentials:
                                "include"
                        }
                    );

                }
                finally {

                    clearAccessToken();

                    authForm.reset();

                    onLoggedOut();

                }

            }
        );

}