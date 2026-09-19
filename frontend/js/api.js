export const API_BASE_URL =
    window.location.origin;


const ACCESS_TOKEN_KEY =
    "careerai_access_token";


export function getAccessToken() {

    return localStorage.getItem(
        ACCESS_TOKEN_KEY
    );

}


export function setAccessToken(token) {

    localStorage.setItem(
        ACCESS_TOKEN_KEY,
        token
    );

}


export function clearAccessToken() {

    localStorage.removeItem(
        ACCESS_TOKEN_KEY
    );

}


async function refreshAccessToken() {

    const response = await fetch(
        `${API_BASE_URL}/auth/refresh`,
        {
            method: "POST",
            credentials: "include"
        }
    );

    if (!response.ok) {

        clearAccessToken();

        return false;

    }

    const data = await response.json();

    setAccessToken(
        data.access_token
    );

    return true;

}


export async function apiFetch(
    path,
    options = {},
    retry = true
) {

    const token = getAccessToken();

    const headers = new Headers(
        options.headers || {}
    );

    if (token) {

        headers.set(
            "Authorization",
            `Bearer ${token}`
        );

    }


    const response = await fetch(
        `${API_BASE_URL}${path}`,
        {
            ...options,
            headers,
            credentials: "include"
        }
    );


    if (
        response.status === 401
        && retry
    ) {

        const refreshed =
            await refreshAccessToken();

        if (refreshed) {

            return apiFetch(
                path,
                options,
                false
            );

        }

    }


    return response;

}