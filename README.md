# 🐾 ArkIve — Motor de Inteligência Clínica Veterinária

> **FIAP Challenge 2026 — 2º Ano ADS | Turmas de Fevereiro**  
> Parceria: **Clyvo Vet** · Disciplina: *Disruptive Architectures: IoT, IoB & Generative AI*

---

## 👥 Equipe

| Nome | RM |
|------|----|
| Gustavo Crevelari | RM561408 |
| Lucca Gomes | RM561996 |
| Rafaela Ferreira | RM561671 |
| Victor Sabelli | RM566224 |

---

## 🎯 Problema Abordado

A jornada de saúde do pet é **fragmentada e reativa**. Responsáveis e veterinários interagem apenas em momentos pontuais — vacinas, emergências, retornos — sem continuidade inteligente entre as consultas.

Do ponto de vista clínico, isso significa que o veterinário frequentemente:

- Não tem acesso rápido ao histórico consolidado do animal no momento da consulta;
- Precisa cruzar manualmente sintomas, raça, espécie e predisposições genéticas;
- Toma decisões sem suporte de evidências clínicas atualizadas.

**Impacto direto:** agravamento evitável de quadros, baixa adesão a tratamentos e perda de recorrência para as clínicas.

---

## 💡 Solução Proposta

O **Motor de Inteligência Clínica Veterinária ArkIve** é um microsserviço Python que, a partir do ID de uma consulta veterinária, extrai automaticamente os dados clínicos do banco Oracle (histórico do animal, sintomas, bem-estar, predisposições genéticas da raça), e aciona um modelo de linguagem (LLM) via API para gerar uma **hipótese diagnóstica estruturada** — pronta para ser consumida pelo serviço Java que persiste o resultado no banco.

O sistema opera em **modo estritamente read-only** no banco, nunca escrevendo ou alterando dados.

### Como a solução melhora a jornada

| Para quem | Benefício |
|-----------|-----------|
| **Pet** | Hipótese diagnóstica mais fundamentada, considerando predisposição genética e histórico clínico |
| **Responsável** | Continuidade do cuidado — cada consulta alimenta a inteligência do sistema |
| **Veterinário** | Apoio à decisão clínica em segundos, com raciocínio explicado e grau de confiança |
| **Clínica** | Diferencial competitivo com IA integrada ao prontuário existente |

---

## 🧠 Tecnologias Utilizadas

| Camada | Tecnologia |
|--------|-----------|
| Linguagem | Python 3.11+ |
| LLM / IA Generativa | [Groq API](https://console.groq.com) · modelo `llama-3.3-70b-versatile` (free tier) |
| Orquestração LLM | LangChain + `with_structured_output()` |
| Banco de Dados | Oracle (modo Thin via `oracledb`) |
| Validação de Schema | Pydantic v2 |
| Busca Web (fallback) | DuckDuckGo Search (`duckduckgo-search`) |
| Variáveis de Ambiente | `python-dotenv` |

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                      main.py (entrada)                          │
│                   python main.py <ID_CONSULTA>                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              ClinicalIntelligenceEngine                         │
│                  agents/clinical_agent.py                       │
│                                                                 │
│  Etapa 1 ──► Oracle (READ-ONLY)                                 │
│              Extrai: animal, espécie, raça, consulta,           │
│              bem-estar, predisposições genéticas                │
│                                                                 │
│  Etapa 2 ──► Heurística local (Python puro, zero API)           │
│              Score de qualidade dos dados clínicos              │
│              Decide se busca web é necessária                   │
│                                                                 │
│  Etapa 3 ──► DuckDuckGo (condicional, zero API)                 │
│              Busca literatura veterinária se score < 60%        │
│                                                                 │
│  Etapa 4 ──► Groq API (UMA única chamada)                       │
│              LLaMA 3.3 70B gera DiagnosticoOutput               │
│              validado pelo Pydantic v2                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    JSON de Saída                                 │
│   {ds_diagnostico, tp_severidade, ds_insight_ia,               │
│    pc_confianca, fontes_pesquisadas}                            │
│                                                                 │
│   → Consumido pelo serviço Java para persistir em               │
│     TB_ARKIVE_DIAGNOSTICO                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Estrutura de Arquivos

```
arkive_clinical_engine/
├── .env.example              # Template de variáveis de ambiente
├── requirements.txt          # Dependências com versões fixas
├── config.py                 # Configuração centralizada + validação fail-fast
├── main.py                   # Ponto de entrada CLI
├── agents/
│   └── clinical_agent.py     # Motor principal (LangChain + Groq + heurística)
├── database/
│   ├── connection.py         # Conexão Oracle Thin mode, READ-ONLY
│   └── queries.py            # SQLs parametrizados + dataclass ClinicalContext
└── schemas/
    └── diagnostic.py         # Pydantic v2: DiagnosticoOutput
```

---

## ⚙️ Como Executar (How To)

### Pré-requisitos

- Python 3.11 ou superior
- Acesso ao banco Oracle da FIAP (`oracle.fiap.com.br`)
- Conta gratuita no [Groq Console](https://console.groq.com) para obter a API Key

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/arkive_clinical_engine.git
cd arkive_clinical_engine
```

### 2. Criar e ativar o ambiente virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

> **Atenção — Windows:** se ocorrer erro de compilação C++ ao instalar `oracledb`, instale o [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) ou tente:
> ```bash
> pip install oracledb==2.3.0 --only-binary=:all:
> ```

### 4. Configurar as variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env`:

```env
# Banco Oracle FIAP
ORACLE_DSN=oracle.fiap.com.br:1521/ORCL
ORACLE_USER=seu_usuario_fiap
ORACLE_PASSWORD=sua_senha_fiap

# Groq (obtenha em https://console.groq.com → API Keys)
GROQ_API_KEY=gsk_sua_chave_aqui

# Configurações opcionais
LOG_LEVEL=INFO
AMBIGUITY_THRESHOLD=60
```

### 5. Executar

```bash
python main.py <ID_CONSULTA>
```

**Exemplo:**

```bash
python main.py 1
```

### 6. Saída esperada

```json
{
  "ds_diagnostico": "Suspeita de Gastroenterite Infecciosa Canina",
  "tp_severidade": "MODERADA",
  "ds_insight_ia": "O paciente Rex apresenta vômito frequente e fezes moles...",
  "pc_confianca": 72,
  "fontes_pesquisadas": []
}
```

Logs de execução são exibidos no terminal. Em caso de erro, o JSON de saída conterá o campo `"error"` com a causa.

---

## 🔄 Fluxo de Decisão da IA

O sistema avalia a qualidade dos dados clínicos localmente **antes** de acionar a LLM, usando uma heurística sem custo de API:

| Critério | Penalidade no Score |
|----------|-------------------|
| Sintomas ausentes ou < 20 caracteres | -35 pts |
| Motivo da consulta vazio | -20 pts |
| Espécie exótica (não cão/gato) | -20 pts |
| Sem predisposições genéticas mapeadas | -10 pts |
| Avaliação de bem-estar ausente | -10 pts |
| Sintomas genéricos (1 palavra) | -15 pts |

Se o score ficar **abaixo de 60%** (configurável em `AMBIGUITY_THRESHOLD`), o DuckDuckGo é acionado para buscar literatura veterinária atualizada, e o contexto é enriquecido antes da chamada à LLM.

**Resultado: NO MÁXIMO 1 chamada à API do Groq por execução.**

---

## 📊 Schema de Saída (Pydantic v2)

Mapeado para gravação futura na tabela `TB_ARKIVE_DIAGNOSTICO` pelo serviço Java:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `ds_diagnostico` | `str` (5–500 chars) | Título conciso da hipótese diagnóstica |
| `tp_severidade` | `Literal["LEVE", "MODERADA", "GRAVE"]` | Classificação de severidade |
| `ds_insight_ia` | `str` (mín. 50 chars) | Raciocínio clínico detalhado da IA |
| `pc_confianca` | `int` (0–100) | Grau de certeza estimado |
| `fontes_pesquisadas` | `list[str]` | URLs consultadas (vazia se busca web não foi acionada) |

---

## 🛡️ Garantias de Segurança (READ-ONLY)

O microsserviço garante a imutabilidade do banco em três camadas:

1. **Privilégios DB:** o usuário Oracle deve ter apenas `GRANT SELECT` nas tabelas ArkIve (enforçado pelo DBA);
2. **`autocommit = False`:** configurado explicitamente na conexão;
3. **`rollback()` no finally:** desfaz qualquer transação pendente acidental antes de fechar a conexão.

Nenhum `INSERT`, `UPDATE`, `DELETE` ou `MERGE` existe em qualquer arquivo do projeto.

---

## 🔧 Solução de Problemas Comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `ORA-00932: inconsistent datatypes` | `SELECT DISTINCT` em coluna CLOB | Já corrigido na versão atual via subquery |
| `ORA-12505` | SID não reconhecido | Usar o DSN no formato longo: `(DESCRIPTION=(ADDRESS=...)(CONNECT_DATA=(SID=ORCL)))` |
| `ORA-01017` | Usuário/senha incorretos | Verificar `ORACLE_USER` e `ORACLE_PASSWORD` no `.env` |
| `429 quota exceeded` | Cota diária da API atingida | A cota do Groq é de ~14.400 req/dia; aguardar reset à meia-noite ou criar nova API key |
| `ModuleNotFoundError` | Dependência não instalada | Rodar `pip install -r requirements.txt` com o venv ativo |
| `Nenhuma consulta encontrada` | ID inexistente no banco | Verificar se o ID existe em `TB_ARKIVE_CONSULTA` |

---

## 📋 Dependências

```
oracledb>=2.3.0,<3.0.0
langchain>=0.3.0,<0.4.0
langchain-groq>=0.2.0,<1.0.0
langchain-community>=0.3.0,<0.4.0
duckduckgo-search>=6.2.0,<7.0.0
pydantic>=2.7.0,<3.0.0
python-dotenv>=1.0.0,<2.0.0
```

---

## 📄 Licença

Projeto acadêmico desenvolvido para o Challenge FIAP 2026 em parceria com a Clyvo Vet.  
Uso restrito ao contexto educacional.