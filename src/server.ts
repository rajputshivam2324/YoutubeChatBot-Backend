import { fetchTranscript } from 'youtube-transcript-plus';
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync } from 'fs';

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
  origin: true, // Allow all origins
  credentials: true
}))
app.use(express.json({ limit: '50mb' }))


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

app.post('/ytchatbot',async(req,res)=>{
    try {
        // Validate input
        const videoUrl = req.body?.videoUrl;
        const question = req.body?.question;
        
        if (!videoUrl || !question) {
            return res.status(400).json({ 
                error: 'Missing required fields: videoUrl and question are required' 
            });
        }

        // Extract video ID to create unique collection name
        const videoId = extractVideoId(videoUrl);
        const collectionName = `ytchatbot_${videoId}`;
        console.log('Using collection name:', collectionName, 'for video:', videoUrl);

        console.log('Fetching transcript for:', videoUrl);
        const transcript = await fetchTranscript(videoUrl, { lang: 'en' });
        
        if (!transcript || transcript.length === 0) {
            return res.status(400).json({ 
                error: 'Could not fetch transcript for this video. Please check if the video has captions enabled.' 
            });
        }

        console.log('Creating Chroma client...');
        const client = new CloudClient({
            apiKey: process.env.chroma,
            tenant: process.env.tenant,
            database: 'langchain',
        });

        const fullText = transcript.map(item => item.text).join(' ');

        console.log('Splitting text into chunks...');
        const textSplitter = new CharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
        const chunks = await textSplitter.splitText(fullText);
        
        console.log('Creating embeddings...');
        const embeddings = new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GOOGLE_API_KEY,
            model: "models/text-embedding-004",
            taskType: TaskType.RETRIEVAL_DOCUMENT,
        });
        
        console.log('Creating vector store with unique collection:', collectionName);
        // Use unique collection name per video to avoid mixing transcripts
        const vectorStore = await Chroma.fromTexts(
            chunks,
            chunks.map((_, i) => ({ source: "youtube", chunkIndex: i, videoId: videoId })),
            embeddings,
            {
                collectionName: collectionName,
                index: client as any, 
            }
        );
        
        console.log('Setting up retriever and LLM...');
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
            ["system",`You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.`
            ],
            ["human", "Question: {question}\n\nContext:\n{context}"]
        ]);
        
        const parser = new StringOutputParser();
        const chain1 = RunnableSequence.from([
            RunnableLambda.from((input: {question: string}) => input.question),
            retriever,
            RunnableLambda.from(format_docs)
        ]);
        
        const parallel_chain = RunnableParallel.from({
            context: chain1,
            question: RunnableLambda.from((input: {question: string}) => input.question),
        });

        const main_chain = RunnableSequence.from([
            parallel_chain,
            promptTemplate,
            llm,
            parser
        ]);

        console.log('Invoking chain with question:', question);
        const ans = await main_chain.invoke({
            question: question
        });
        
        console.log('Successfully generated answer');
        res.json(ans);
    } catch (error: any) {
        console.error('Error in /ytchatbot endpoint:', error);
        res.status(500).json({ 
            error: 'Internal server error',
            message: error?.message || 'An unexpected error occurred',
            details: process.env.NODE_ENV === 'development' ? error?.stack : undefined
        });
    }
});
const PORT = process.env.PORT || 3005;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
})