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

async function fetchTranscriptWithFallback(videoIdentifier: string) {
    for (const lang of TRANSCRIPT_LANGUAGES) {
        try {
            const transcript = await fetchTranscript(videoIdentifier, { lang });
            if (transcript?.length) {
                return transcript;
            }
        } catch (error) {
            console.warn(`Transcript fetch failed for lang ${lang}:`, (error as Error)?.message);
        }
    }

    try {
        const transcript = await fetchTranscript(videoIdentifier);
        if (transcript?.length) {
            return transcript;
        }
    } catch (error) {
        console.warn('Transcript fetch without lang failed:', (error as Error)?.message);
    }

    return [];
}

async function collectionExists(client: CloudClient, collectionName: string) {
    const collections = await client.listCollections();
    return collections.some((col: any) => col.name === collectionName);
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

    for (const identifier of identifiersToTry) {
        transcript = await fetchTranscriptWithFallback(identifier);
        if (transcript?.length) {
            break;
        }
    }

    if (!transcript || transcript.length === 0) {
        throw new HttpError('Could not fetch transcript. Please check if the video has captions enabled.', 404);
    }

    const fullText = transcript.map(item => item.text).join(' ');
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
            vectorStore = await createVectorStoreForVideo({
                videoUrl,
                videoId,
                transcriptId,
                collectionName,
                embeddings,
                client,
            });
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
app.listen(3001);