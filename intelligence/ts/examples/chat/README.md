# Next.js Chat App with FlowerIntelligence

A minimalistic chat application built with Next.js, TypeScript, and Tailwind CSS. This project streams AI responses using the FlowerIntelligence library, supports multiple models with selectable rendering, and includes remote handoff with API key support.

## Features

- **Minimalistic UI:** Clean chat interface with user and AI messages.
- **Real-time Streaming:** AI responses stream in as they are generated.
- **Model Selection:** Choose from several AI models using a dropdown.
- **Remote Handoff:** Enable remote processing via a toggle (API key loaded from environment variables).
- **Internal Reasoning Display:** For deepseek, display the AI’s internal reasoning in a collapsible panel.
- **Model Loading:** Fetch models on first use and display a detailed loading description.

## Setup

1. Install: `npm install`
2. Copy the example environment file and update it with your API key: `cp .env.example .env`

3. Edit the **.env** file so it contains: `NEXT_PUBLIC_API_KEY=your-api-key-here`
4. Run the project: `npm run dev`

5. Open http://localhost:3000 in your browser.

## Project Structure

- **app/layout.tsx:** Root layout with global styles and background image.
- **app/page.tsx:** Main chat interface with model selection, remote handoff, streaming responses, and collapsible internal reasoning.
- **styles/globals.css:** Global CSS including Tailwind directives.
- **.env.example:** Example file for environment variables.

## How to Use

- **Chat Interface:**Type your question in the input field and press Enter or click the Send button.
- **Model Selection:**Use the dropdown to choose an AI model. Each model may render responses differently.
- **Remote Handoff:**Toggle "Allow remote handoff" to enable remote processing (the API key is loaded from the environment).
- **Model Loading:**When a model is used for the first time, it is fetched while displaying a descriptive loading status until complete.
- **Response Rendering:**AI responses are streamed in real time. For the deepseek model, internal reasoning wrapped in ... is extracted and shown in a collapsible panel (closed by default).

## Additional Information

- **Dependencies:**

  - Next.js
  - Tailwind CSS
  - FlowerIntelligence
  - React Markdown
  - Remark GFM

- **Deployment:**For production deployment, refer to the Next.js documentation.
- **License:**This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

Enjoy your chat app!
