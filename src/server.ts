import { fetchTranscript } from 'youtube-transcript-plus';
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync } from 'fs';
import { randomUUID } from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env from backend2 directory
// Try parent directory first (when running from src/), then try parent of parent (when running from dist/)
const envPath1 = join(__dirname, '../.env');
const envPath2 = join(__dirname, '../../.env');
const envPath = existsSync(envPath1) ? envPath1 : envPath2;

dotenv.config({ path: envPath });
import { CloudClient } from "chromadb";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers"
import { RunnableLambda, RunnableParallel, RunnableSequence } from '@langchain/core/runnables';
import { Document } from "@langchain/core/documents";

import express from 'express'
import cors from 'cors'

const app= express()
// CORS configuration - allow all origins for flexibility
app.use(cors({
  origin: [
    'https://chatifyai.vercel.app',
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}))
app.use(express.json({ limit: '50mb' }))

// Health check endpoint for Render
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});


class HttpError extends Error {
    statusCode: number;

    constructor(message: string, statusCode = 500) {
        super(message);
        this.statusCode = statusCode;
    }
}

const TRANSCRIPT_LANGUAGES = ['en', 'en-US', 'en-GB', 'en-IN', 'en-CA', 'en-AU'];
const TRANSCRIPT_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;
const YOUTUBE_VIDEO_ID_PATTERN = /^[a-zA-Z0-9_-]{11}$/;
const COLLECTION_PREFIX = 'ytchatbot_';

const buildCollectionName = (videoId: string, transcriptId: string) => `${COLLECTION_PREFIX}${videoId}_${transcriptId}`;

// Helper function to add timeout to promises
function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    return Promise.race([
        promise,
        new Promise<T>((_, reject) => 
            setTimeout(() => reject(new Error(`Operation timed out after ${timeoutMs}ms`)), timeoutMs)
        )
    ]);
}

async function fetchTranscriptWithFallback(videoIdentifier: string, retries = 3) {
    const errors: string[] = [];
    const TIMEOUT_MS = 30000; // 30 seconds timeout
    
    // Helper function to retry with delay
    const retryWithDelay = async (fn: () => Promise<any>, delay: number, attempt: number): Promise<any> => {
        try {
            return await withTimeout(fn(), TIMEOUT_MS);
        } catch (error: any) {
            if (attempt < retries) {
                console.log(`Retry attempt ${attempt + 1}/${retries} after ${delay}ms delay`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return retryWithDelay(fn, delay * 1.5, attempt + 1);
            }
            throw error;
        }
    };

    // Try with specific languages first
    for (const lang of TRANSCRIPT_LANGUAGES) {
        try {
            const transcript = await retryWithDelay(
                () => fetchTranscript(videoIdentifier, { lang }),
                1000,
                0
            );
            if (transcript?.length) {
                console.log(`Successfully fetched transcript with lang: ${lang}`);
                return transcript;
            }
        } catch (error: any) {
            const errorMsg = error?.message || String(error);
            errors.push(`Lang ${lang}: ${errorMsg}`);
            console.warn(`Transcript fetch failed for lang ${lang}:`, errorMsg);
        }
    }

    // Try without language specification
    try {
        const transcript = await retryWithDelay(
            () => fetchTranscript(videoIdentifier),
            1000,
            0
        );
        if (transcript?.length) {
            console.log('Successfully fetched transcript without lang specification');
            return transcript;
        }
    } catch (error: any) {
        const errorMsg = error?.message || String(error);
        errors.push(`No lang: ${errorMsg}`);
        console.warn('Transcript fetch without lang failed:', errorMsg);
    }

    // Log all errors for debugging
    console.error('All transcript fetch attempts failed:', errors);
    return [];
}

async function collectionExists(client: CloudClient, collectionName: string): Promise<boolean> {
    try {
        // Try to get the collection directly instead of listing all collections
        // This avoids warnings from other collections in the database
        const collection = await client.getCollection({ name: collectionName });
        return collection !== null && collection !== undefined;
    } catch (error: any) {
        // If collection doesn't exist, getCollection will throw an error
        // Check if it's a "not found" type error
        if (error?.message?.includes('not found') || error?.message?.includes('does not exist')) {
            return false;
        }
        // For other errors, fall back to listing (but this may trigger warnings)
        try {
            const collections = await client.listCollections();
            return collections.some((col: any) => col.name === collectionName);
        } catch {
            return false;
        }
    }
}

type CreateVectorStoreParams = {
    videoUrl: string;
    videoId: string;
    transcriptId: string;
    collectionName: string;
    embeddings: GoogleGenerativeAIEmbeddings;
    client: CloudClient;
};

async function createVectorStoreForVideo({
    videoUrl,
    videoId,
    transcriptId,
    collectionName,
    embeddings,
    client,
}: CreateVectorStoreParams): Promise<Chroma> {
    const identifiersToTry = YOUTUBE_VIDEO_ID_PATTERN.test(videoId)
        ? [videoId, videoUrl]
        : [videoUrl, videoId];

    let transcript = [] as Awaited<ReturnType<typeof fetchTranscriptWithFallback>>;
    let lastError: Error | null = null;

    for (const identifier of identifiersToTry) {
        try {
            console.log(`Attempting to fetch transcript for identifier: ${identifier}`);
            transcript = await fetchTranscriptWithFallback(identifier);
            if (transcript?.length) {
                console.log(`Successfully fetched transcript with ${transcript.length} items`);
                break;
            }
        } catch (error: any) {
            lastError = error;
            console.error(`Failed to fetch transcript for ${identifier}:`, error?.message || error);
        }
    }

    if (!transcript || transcript.length === 0) {
        const errorMessage = lastError 
            ? `Could not fetch transcript: ${lastError.message}. Please check if the video has captions enabled and is accessible.`
            : 'Could not fetch transcript. Please check if the video has captions enabled.';
        console.error('Transcript fetch failed:', errorMessage);
        throw new HttpError(errorMessage, 404);
    }

    const fullText = transcript.map((item: { text: string }) => item.text).join(' ');
    const textSplitter = new CharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const chunks = await textSplitter.splitText(fullText);

    if (!chunks.length) {
        throw new HttpError('Transcript is empty after processing. Try another video.', 422);
    }

    return await Chroma.fromTexts(
        chunks,
        chunks.map((_, i) => ({ source: "youtube", chunkIndex: i, videoId, transcriptId })),
        embeddings,
        {
            collectionName,
            index: client as any,
        }
    );
}


// Helper function to extract video ID from YouTube URL
function extractVideoId(url: string): string {
    const patterns = [
        /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
        /youtube\.com\/watch\?.*v=([^&\n?#]+)/
    ];
    
    for (const pattern of patterns) {
        const match = url.match(pattern);
        if (match && match[1]) {
            return match[1];
        }
    }
    
    // If no match, create a hash from the URL
    return Buffer.from(url).toString('base64').replace(/[^a-zA-Z0-9]/g, '').substring(0, 20);
}

app.post('/ytchatbot', async (req, res) => {
    try {
        const videoUrl = typeof req.body?.videoUrl === 'string' ? req.body.videoUrl.trim() : '';
        const question = typeof req.body?.question === 'string' ? req.body.question.trim() : '';
        let transcriptId = typeof req.body?.transcriptId === 'string' ? req.body.transcriptId.trim() : '';

        if (!videoUrl || !question) {
            return res.status(400).json({
                error: 'Missing required fields: videoUrl and question are required'
            });
        }

        if (transcriptId && !TRANSCRIPT_ID_PATTERN.test(transcriptId)) {
            return res.status(400).json({
                error: 'Invalid transcriptId format. Only alphanumeric, hyphen, and underscore are allowed.'
            });
        }

        // Extract video ID - each video gets unique ID
        const videoId = extractVideoId(videoUrl);
        const isNewTranscript = !transcriptId;

        if (!transcriptId) {
            transcriptId = randomUUID();
        }

        const collectionName = buildCollectionName(videoId, transcriptId);

        console.log('Video ID:', videoId, '| Transcript ID:', transcriptId, '| Collection:', collectionName);

        // Create Chroma client
        const client = new CloudClient({
            apiKey: process.env.chroma,
            tenant: process.env.tenant,
            database: 'langchain',
        });

        const embeddings = new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GOOGLE_API_KEY,
            model: "models/text-embedding-004",
            taskType: TaskType.RETRIEVAL_DOCUMENT,
        });

        let vectorStore: Chroma;
        const collectionAlreadyExists = await collectionExists(client, collectionName);

        if (collectionAlreadyExists) {
            console.log('Using existing collection for transcript ID:', transcriptId);
            vectorStore = await Chroma.fromExistingCollection(embeddings, {
                collectionName,
                index: client as any,
            });
        } else {
            console.log('Creating new collection for transcript ID:', transcriptId);
            console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
            console.log(`Video URL: ${videoUrl}, Video ID: ${videoId}`);
            try {
                vectorStore = await createVectorStoreForVideo({
                    videoUrl,
                    videoId,
                    transcriptId,
                    collectionName,
                    embeddings,
                    client,
                });
            } catch (error: any) {
                console.error('Error creating vector store:', {
                    message: error?.message,
                    stack: error?.stack,
                    videoUrl,
                    videoId
                });
                throw error;
            }
        }

        // Setup retriever and LLM
        const retriever = vectorStore.asRetriever();
        const llm = new ChatGoogleGenerativeAI({
            model: "gemini-2.5-flash",
            apiKey: process.env.GOOGLE_API_KEY,
            temperature: 0.7,
        });

        const format_docs = (retrievedDocs: Document[]) => {
            return retrievedDocs.map(doc => doc.pageContent).join("\n\n");
        }

        const promptTemplate = ChatPromptTemplate.fromMessages([
            ["system", `You are a helpful assistant. Answer ONLY from the provided transcript context. If the context is insufficient, just say you don't know.`],
            ["human", "Question: {question}\n\nContext:\n{context}"]
        ]);

        const parser = new StringOutputParser();
        const chain1 = RunnableSequence.from([
            RunnableLambda.from((input: { question: string }) => input.question),
            retriever,
            RunnableLambda.from(format_docs)
        ]);

        const parallel_chain = RunnableParallel.from({
            context: chain1,
            question: RunnableLambda.from((input: { question: string }) => input.question),
        });

        const main_chain = RunnableSequence.from([
            parallel_chain,
            promptTemplate,
            llm,
            parser
        ]);

        const answer = await main_chain.invoke({ question });
        res.json({
            answer,
            transcriptId,
            isNewTranscript: isNewTranscript || !collectionAlreadyExists,
        });
    } catch (error: any) {
        if (error instanceof HttpError) {
            return res.status(error.statusCode).json({ error: error.message });
        }

        console.error('Error in /ytchatbot endpoint:', error);
        res.status(500).json({
            error: 'Internal server error',
            message: error?.message || 'An unexpected error occurred',
            details: process.env.NODE_ENV === 'development' ? error?.stack : undefined
        });
    }
});
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});