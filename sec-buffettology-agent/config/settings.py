import os

# Intentar importar dotenv de forma segura
try:
    from dotenv import load_dotenv
    # Cargar variables de entorno desde el archivo .env si existe
    load_dotenv()
except ImportError:
    print("[AVISO] La librería 'python-dotenv' no está disponible en este entorno. Se omitirá.")

# Variables de entorno críticas (con valores por defecto seguros para el IDE web)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "TU_API_KEY_DE_PRUEBA")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "UsuarioPrueba usuario@ejemplo.com")

if GEMINI_API_KEY == "TU_API_KEY_DE_PRUEBA":
    print("[ADVERTENCIA] GEMINI_API_KEY no configurada. Usando clave de prueba (fallará en llamadas reales).")
if SEC_USER_AGENT == "UsuarioPrueba usuario@ejemplo.com":
    print("[ADVERTENCIA] SEC_USER_AGENT no configurado. La SEC podría bloquear las peticiones.")

# Rutas del sistema
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache_sec")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output_reports")
NORMALIZED_DIR = os.path.join(BASE_DIR, "data", "normalized_json")
PROMPTS_DIR = os.path.join(BASE_DIR, "config", "prompts")

# Crear estructura de carpetas de datos si no existe
for directory in [CACHE_DIR, OUTPUT_DIR, NORMALIZED_DIR, PROMPTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuración por defecto de Gemini
DEFAULT_MODEL = "gemini-1.5-pro"  # Usamos la versión estable más reciente