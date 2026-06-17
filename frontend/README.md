# Frontend — Decision Console (Next.js)

Frontend **opcional** (bônus de portfólio) que consome a **FastAPI** do projeto.
O núcleo de ML/MLOps continua 100% Python — este app é só a camada de apresentação.

## Stack
- **Next.js 15** (App Router) + **React 19** + **TypeScript**
- **Tailwind CSS v4** (design tokens em `app/globals.css`)
- **Radix UI** (Slider, Select acessíveis)
- **Framer Motion** (animações)
- **Lucide Icons**
- **Recharts** (gráfico de valor por oferta)

## Como rodar
Pré-requisito: a **API Python rodando** em `http://localhost:8000`:
```powershell
# terminal 1 (na raiz do projeto Python, com o venv ativo)
adaptive-offers serve
```
```powershell
# terminal 2 (nesta pasta frontend/)
npm install         # ou: pnpm install
npm run dev         # abre http://localhost:3000
```
O `next.config.mjs` faz proxy de `/api/*` → `http://localhost:8000/*` (sem CORS;
a API Python não é alterada). Para apontar para outra URL: `API_URL=... npm run dev`.

## O que mostra
- Status da política/API (`/health`, `/policy`)
- KPIs da política ativa (reward/1k, regret, conversão, exploração)
- **Explorador de decisão**: ajusta o contexto → `/decide` + `/assistant/explain` →
  oferta escolhida, reason codes, **gráfico de valor por oferta** (Recharts) e
  explicação do assistente (LLM/RAG)
- Catálogo de ofertas (`/offers`)

## Build de produção
```powershell
npm run build && npm run start
```
