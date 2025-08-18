/// <reference types="vite/client" />

interface ImportMetaEnv {
	readonly VITE_API_URL: string
	readonly VITE_WS_URL: string
	readonly VITE_SOLANA_RPC?: string
}

interface ImportMeta {
	readonly env: ImportMetaEnv
}