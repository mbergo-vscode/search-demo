import { GoogleGenAI } from "@google/genai";

export const generateRAGResponse = async (query: string, context: string): Promise<string> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const model = 'gemini-3-flash-preview';
    const prompt = `
      You are an AI assistant in a RAG pipeline simulation.
      
      Context retrieved from Vector DB:
      ${context}
      
      User Query: ${query}
      
      Please provide a concise answer (max 2 sentences) based strictly on the context.
    `;

    const response = await ai.models.generateContent({
      model: model,
      contents: prompt,
    });
    
    return response.text || "No response generated.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Error generating response from Gemini. Check console.";
  }
};

export const generateImage = async (prompt: string, size: '1K' | '2K' | '4K'): Promise<string | null> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-image-preview',
      contents: {
        parts: [
          {
            text: prompt,
          },
        ],
      },
      config: {
        imageConfig: {
          aspectRatio: "1:1",
          imageSize: size
        },
      },
    });

    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData && part.inlineData.data) {
        const base64EncodeString: string = part.inlineData.data;
        return `data:image/png;base64,${base64EncodeString}`;
      }
    }
    return null;
  } catch (error) {
    console.error("Gemini Image Gen Error:", error);
    throw error;
  }
};

export const generateVeoVideo = async (
    prompt: string, 
    imageBase64?: string, 
    aspectRatio: '16:9' | '9:16' = '16:9'
): Promise<string | null> => {
    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const request: any = {
            model: 'veo-3.1-fast-generate-preview',
            prompt: prompt,
            config: {
                numberOfVideos: 1,
                resolution: '720p',
                aspectRatio: aspectRatio
            }
        };

        if (imageBase64) {
            request.image = {
                imageBytes: imageBase64,
                mimeType: 'image/png',
            };
        }

        // Initial Request
        let operation = await ai.models.generateVideos(request);

        // Polling loop
        while (!operation.done) {
            await new Promise(resolve => setTimeout(resolve, 5000)); // Poll every 5s
            operation = await ai.operations.getVideosOperation({operation: operation});
        }

        const videoUri = operation.response?.generatedVideos?.[0]?.video?.uri;
        if (!videoUri) return null;

        // Fetch the actual bytes using the key
        const videoResponse = await fetch(`${videoUri}&key=${process.env.API_KEY}`);
        const videoBlob = await videoResponse.blob();
        return URL.createObjectURL(videoBlob);

    } catch (error) {
        console.error("Veo Video Gen Error:", error);
        throw error;
    }
};