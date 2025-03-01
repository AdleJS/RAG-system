import "cheerio";
import { config } from "dotenv";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

config();

// initialization
const llm = new ChatOllama({
    model: "llama3.2:3b",
    temperature: 0,
});


const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text:latest", // Default value
    baseUrl: "http://localhost:11434", // Default value
});

const vectorStore = new MemoryVectorStore(embeddings);
const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/", { selector: pTagSelector });

// chunking documents
const docs = await cheerioLoader.load();
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
const allSplits = await splitter.splitDocuments(docs);

// indexing chunks
await vectorStore.addDocuments(allSplits);

// define prompt template
const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

// app state
const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>
});

const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<Document[]>,
    answer: Annotation<string>,
});

const retrieve = async (state: typeof InputStateAnnotation.State) => {
    const retrievedDocs = await vectorStore.similaritySearch(state.question)
    return { context: retrievedDocs };
};

const generate = async (state: typeof StateAnnotation.State) => {
    const docsContent = state.context.map(doc => doc.pageContent).join("\n");
    const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
    const response = await llm.invoke(messages);
    return { answer: response.content };
};

const graph = new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();


let inputs = { question: "What is Task Decomposition?" };

const result = await graph.invoke(inputs);
console.log(result.answer);