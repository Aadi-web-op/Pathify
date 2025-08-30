// src/api/gemini.js

/**
 * A reusable function to call the Google Gemini API.
 * @param {string} prompt - The text prompt to send to the AI.
 * @param {boolean} isJson - Set to true if you expect a JSON formatted string in response.
 * @returns {Promise<string>} - The text response from the AI.
 */
export async function callGeminiAPI(prompt, isJson = false) {
    // IMPORTANT: In a real production app, you should never expose your API key like this.
    // Use environment variables to keep it secure. For this project, you can get a key
    // from Google AI Studio and paste it here.
    const apiKey = "***REMOVED***"; // <-- PASTE YOUR GEMINI API KEY HERE

    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
    
    const payload = {
        contents: [{ parts: [{ text: prompt }] }],
    };

    // If we expect a JSON response, we modify the payload to ask the model for it.
    if (isJson) {
        payload.generationConfig = { responseMimeType: "application/json" };
    }

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            console.error("API Error Response:", await response.text());
            throw new Error(`API request failed with status ${response.status}`);
        }

        const result = await response.json();
        // Navigate through the response object to find the generated text.
        const text = result.candidates?.[0]?.content?.parts?.[0]?.text;

        if (!text) {
            console.error("No text in API response:", result);
            throw new Error("Invalid response from API");
        }
        return text;
    } catch (error) {
        console.error("Error calling Gemini API:", error);
        throw error; // Rethrow the error so the calling component can handle it.
    }
}
