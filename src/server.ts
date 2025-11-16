import { fetchTranscript } from 'youtube-transcript-plus';
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import dotenv from 'dotenv';
dotenv.config();
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

app.use(cors({
  origin: 'https://chatifyai.vercel.app',
  credentials: true
}));


app.use(express.json())

// Health check endpoint for Render
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

app.post('/ytchatbot',async(req,res)=>{
    const videoUrl= req.body.videoUrl
    const question= req.body.question
    const transcript = await fetchTranscript(videoUrl, { lang: 'en' });
    
    const client = new CloudClient({
        apiKey: process.env.chroma,
        tenant: process.env.tenant,
        database: 'langchain',
    });

    const fullText = transcript.map(item => item.text).join(' ');


    const textSplitter = new CharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    const chunks = await textSplitter.splitText(fullText);
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GOOGLE_API_KEY,
        model: "models/text-embedding-004",
        taskType: TaskType.RETRIEVAL_DOCUMENT,
    });
    const vectorStore = await Chroma.fromTexts(
        chunks,
        chunks.map((_, i) => ({ source: "youtube", chunkIndex: i })),
        embeddings,
        {
            collectionName: "ytchatbot",
            index: client as any, 
        }
    );
    const retriever = vectorStore.asRetriever();
    const llm = new ChatGoogleGenerativeAI({
        model: "gemini-2.5-flash",
        apiKey: process.env.GOOGLE_API_KEY,
        temperature: 0.7,
    });
    const format_docs=(retrievedDocs :Document[])=>{
        return retrievedDocs.map(doc => doc.pageContent).join("\n\n");
    }
    const promptTemplate = ChatPromptTemplate.fromMessages([
        ["system",`You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.`
        ],
        ["human", "Question: {question}\n\nContext:\n{context}"]
    ]);
    const parser= new StringOutputParser()
    const chain1= RunnableSequence.from([
        RunnableLambda.from((input: {question: string}) => input.question),
        retriever,
        RunnableLambda.from(format_docs)
    ]);
    const parallel_chain= RunnableParallel.from({
        context: chain1,
        question: RunnableLambda.from((input: {question: string}) => input.question),
    })

    const main_chain= RunnableSequence.from([
        parallel_chain,
        promptTemplate,
        llm,
        parser
    ]);

    const ans= await main_chain.invoke({
        question: question
    });
    res.json(ans)

});
app.listen(3000)

