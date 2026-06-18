# public/

Arquivos estáticos servidos pelo Next.js.

## Imagem de fundo (hero)
A imagem de fundo é o arquivo **`hero-bg.png`**. Ela é aplicada como background
fixo do `body` em `app/globals.css` (regra `html, body`), esmaecida por um
overlay preto translúcido (~45%) para o conteúdo continuar legível por cima.

Para trocar a imagem, basta substituir `hero-bg.png` (mesmo nome). O dashboard
Streamlit reutiliza esse mesmo arquivo, reduzindo-o automaticamente para um
embed leve.
