The Deep Structure of Curriculum: A Computational Comparative AnalysisThis repository contains the source code, data processing pipeline, and visualization scripts for the research paper: "The Deep Structure of Curriculum: A Computational Comparative Analysis of Chinese, American, and European Undergraduate Education using Large Language Models."📖 AbstractWhile higher education globalization has led to superficial convergence in curriculum goals, the underlying pedagogical structures remain distinct. This study proposes a "Computational Education Sociology" framework to quantify curriculum rigidity, semantic density, and cognitive objectives.Leveraging the BGE-M3 multilingual embedding model and RAG (Retrieval-Augmented Generation), we analyzed core curriculum documents from six top-tier universities (NJU, PKU, Stanford, MIT, Heidelberg, LMU). Our findings reveal a "Goal-Structure Paradox": while cognitive objectives are converging globally, structural typologies diverge significantly between the "Deep Well" model (Strong Framing, typical in China/Europe) and the "Broad Base" model (Weak Framing, typical in the USA).📂 Repository StructureThe project is organized as follows:Curriculum_Analysis/
├── data/
│   ├── raw/                   # Raw text files (University syllabi/handbooks)
│   ├── processed/             # Cleaned JSONL corpus
│   └── embeddings/            # FAISS vector indices and metadata
├── output/
│   ├── figures/               # Generated high-res figures (Main Fig 2, 3, 4)
│   └── tables/                # Statistical tables (CSV)
├── scripts/
│   ├── config.py              # Global configuration and constants
│   ├── 01_data_processing.py  # Data cleaning pipeline
│   ├── 02_build_index.py      # Vector embedding and indexing
│   ├── 03_plot_main_figures.py# Visualization generation (Main Figures)
│   └── 04_generate_tables.py  # Statistical table generation
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
🛠️ InstallationClone the repository:git clone [https://github.com/YourUsername/Curriculum-Analysis.git](https://github.com/YourUsername/Curriculum-Analysis.git)
cd Curriculum-Analysis
Create a virtual environment (Recommended):python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:pip install -r requirements.txt
🚀 Usage PipelineTo reproduce the results presented in the paper, execute the scripts in the following order. All scripts should be run from the root directory.Step 1: Data ProcessingConverts raw curriculum documents into a standardized JSONL format.python scripts/01_data_processing.py
Note: If no raw data is found in data/raw, the script will automatically generate synthetic demo data for testing purposes.Step 2: Vector Index ConstructionGenerates semantic embeddings using BAAI/bge-m3 and builds the FAISS retrieval index.python scripts/02_build_index.py
Step 3: Visualization (Main Figures)Generates the three core composite figures used in the manuscript (Figures 2, 3, and 4), including the Topological Network and AI Diagnosis charts.python scripts/03_plot_main_figures.py
Output: High-resolution TIFF files will be saved in output/figures/.Step 4: Statistical TablesGenerates the supplementary data tables (Corpus stats, Bloom's taxonomy distribution, Network metrics, etc.).python scripts/04_generate_tables.py
Output: CSV files will be saved in output/tables/.📊 Methodology & Key MetricsWe operationalized Basil Bernstein’s code theory into computable metrics:Bernstein's F-Value (Rigidity): Calculated as the ratio of compulsory credits to total credits.Result: LMU/Heidelberg (>0.75) > NJU/PKU (>0.58) > Stanford (0.38).Semantic Density: Measured via pairwise Cosine Similarity of curriculum vectors.Result: NJU exhibits high internal cohesion ("Deep Well"), while Stanford shows high variance ("Broad Base").Cognitive Depth: Quantified using a multilingual lexicon based on Bloom's Taxonomy.Result: Cognitive isomorphism observed across regions, with high emphasis on "Create" and "Analyze".AI Responsiveness: Normalized density of emerging topics (e.g., "AI Ethics") within the curriculum corpus.🛡️ LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
For questions or data access, please open an issue or contact the authors.
