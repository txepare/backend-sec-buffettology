Sistema Multi-Agente de Análisis Financiero SEC (Buffettology)



Este proyecto implementa un sistema de Inteligencia Artificial basado en múltiples agentes impulsados por Google Gemini Pro. Su propósito es extraer datos directamente de la SEC EDGAR, cruzarlos con datos de mercado, aplicar un marco analítico estricto basado en la inversión en valor (Buffettology) y generar un informe final en PDF.



🚀 Instalación y Configuración



Clonar el repositorio y crear el entorno virtual:



python -m venv venv

source venv/bin/activate  # En Windows: venv\\Scripts\\activate





Instalar dependencias:



pip install -r requirements.txt





Configurar credenciales (.env):

Crea un archivo .env en la raíz del proyecto (toma como referencia .env.example) y completa los datos:



GEMINI\_API\_KEY="tu\_api\_key\_de\_gemini"

SEC\_USER\_AGENT="NombreEmpresa tu\_correo@ejemplo.com"





(Nota: El SEC\_USER\_AGENT es obligatorio para evitar bloqueos por parte de la API del gobierno de EE.UU).



🧠 Estructura de Agentes



El sistema opera mediante una canalización orquestada de 6 entidades:



OrchestratorAgent: Coordina el pipeline y el paso de variables.



SecExtractorAgent: Extrae los hechos XBRL (hasta 12 años) desde la API de la SEC.



MarketDataAgent: Extrae cotizaciones en tiempo real mediante Yahoo Finance.



GAAPNormalizerAgent: Estandariza la contabilidad y mapea conceptos XBRL.



SectorConfiguratorAgent: Aplica las reglas sectoriales (Industria, Banca, REITs, Utilities).



ValidationAgent: Audita identidades matemáticas (Ej. Activos = Pasivo + Patrimonio).



PDFGeneratorAgent: Compila el reporte visual utilizando matplotlib y reportlab.



⚙️ Uso



Para iniciar un análisis, ejecuta el archivo principal pasando como argumento el Ticker (Símbolo bursátil) de la empresa:



python main.py AAPL

python main.py JPM





Los reportes PDF generados se guardarán automáticamente en la carpeta data/output\_reports/.

