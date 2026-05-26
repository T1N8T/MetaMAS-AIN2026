import sys
import subprocess
import shutil
import urllib.request
import urllib.error
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import ToolContext
from google.adk.models.lite_llm import LiteLlm
import rag 
import asyncio
from google.adk.runners import InMemoryRunner

load_dotenv()

OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 5
current_retries = 0
best_mas_state = {}
best_error_count = float('inf')


class MASCompletedException(Exception):
    pass


# ==========================================
# DEFINICIÓN DE TOOLS 
# ==========================================

def resolve_jason_command():
    env_path = os.getenv("JASON_BIN")
    if env_path: return env_path
    path_command = shutil.which("jason")
    if path_command: return path_command
    default_macos_path = "/Applications/jason"
    if Path(default_macos_path).exists(): return default_macos_path
    default_windows_paths = [
        r"C:\Jason\bin\jason.bat",
        r"C:\Program Files\Jason\bin\jason.bat",
        r"C:\Program Files (x86)\Jason\bin\jason.bat"
    ]
    for windows_path in default_windows_paths:
        if Path(windows_path).exists(): return windows_path
    return None

def search_github_examples(path: str = "") -> str:
    base_api_url = "https://api.github.com/repos/jason-lang/jason/contents/examples"
    url = f"{base_api_url}/{path}".strip("/")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Python-urllib'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            if isinstance(data, list):
                items = [f"[{item['type']}] {item['path'].replace('examples/', '', 1)}" for item in data]
                return f"Contenido de '{path or 'raíz'}':\n" + "\n".join(items)
            elif isinstance(data, dict) and data.get("type") == "file":
                download_url = data.get("download_url")
                if download_url:
                    req_file = urllib.request.Request(download_url, headers={'User-Agent': 'Python-urllib'})
                    with urllib.request.urlopen(req_file) as f_res:
                        return f_res.read().decode('utf-8')
                return "Error: No se encontró la URL de descarga del archivo."
            else:
                return "Respuesta inesperada de la API de GitHub."
    except Exception as e:
        return f"Error al intentar acceder a los ejemplos: {e}"

def test_mas_code(mas2j_code: str, agents_dict: dict) -> str:
    global current_retries, best_mas_state, best_error_count
    if current_retries >= MAX_RETRIES:
         return f"ERROR: Límite de {MAX_RETRIES} intentos superado."
    current_retries += 1
    temp_dir = Path("temp_mas_project")
    try:
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        mas2j_file = temp_dir / "temp.mas2j"
        mas2j_file.write_text(mas2j_code, encoding="utf-8")
        
        for filename, content in agents_dict.items():
            if not filename.endswith(".asl"): filename += ".asl"
            (temp_dir / filename).write_text(content, encoding="utf-8")
            
        jason_command = resolve_jason_command()
        if not jason_command: return "ERROR: No se ha encontrado Jason."

        result = subprocess.run(
            [jason_command, "mas", "start", "--mas2j=temp.mas2j", "--console"],
            cwd=str(temp_dir), capture_output=True, text=True, timeout=15
        )
        
        error_count = 10 if result.returncode != 0 else 0
        if result.stderr: error_count += len(result.stderr.split('\n'))
            
        if error_count < best_error_count:
            best_error_count = error_count
            best_mas_state = {"mas2j": mas2j_code, "agents": agents_dict}
            
        output = f"=== EJECUCIÓN DE PRUEBA ({current_retries}/{MAX_RETRIES}) ===\n"
        if result.stdout: output += f"--- STDOUT ---\n{result.stdout}\n"
        if result.stderr: output += f"--- STDERR ---\n{result.stderr}\n"
        return output
        
    except subprocess.TimeoutExpired as e:
        stdout_text = e.stdout.decode('utf-8', errors='ignore') if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr_text = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else (e.stderr or "")
        
        output = f"AVISO: Tiempo límite (15s). Normal si el MAS funciona de forma estable.\n"
        if stdout_text: output += f"--- STDOUT ---\n{stdout_text}\n"
        if stderr_text: output += f"--- STDERR ---\n{stderr_text}\n"
        
        if best_error_count == float('inf'):
            best_mas_state = {"mas2j": mas2j_code, "agents": agents_dict}
            
        return output
    except Exception as e:
        return f"ERROR inesperado al ejecutar: {e}"
    finally:
        if temp_dir.exists(): shutil.rmtree(temp_dir)

def save_mas_code(mas_name: str, mas2j_code: str = "", agents_dict: dict = None) -> str:
    global current_retries, best_mas_state, best_error_count
    if agents_dict is None: agents_dict = {}
    project_dir = OUTPUT_DIR / mas_name
    shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    code_mas2j = mas2j_code if mas2j_code else best_mas_state.get("mas2j", "")
    code_agents = agents_dict if agents_dict else best_mas_state.get("agents", {})
    
    if not code_mas2j or not code_agents:
         return "ERROR: No hay código generado para guardar."
         
    try:
        mas_filename = f"{mas_name}.mas2j" if not mas_name.endswith(".mas2j") else mas_name
        (project_dir / mas_filename).write_text(str(code_mas2j), encoding="utf-8")
        for filename, content in code_agents.items():
            if not filename.endswith(".asl"): filename += ".asl"
            (project_dir / filename).write_text(str(content), encoding="utf-8")
        
        current_retries = 0
        best_mas_state = {}
        best_error_count = float('inf')
        return f"ÉXITO: Proyecto BDI guardado en {project_dir}"
    except Exception as e:
        return f"ERROR inesperado al guardar: {e}"


# ==========================================
# INICIALIZACIÓN DE MODELOS
# ==========================================
model = LiteLlm(
    model="openai/Qwen3.6-35B-A3B-FP8",
    api_base=os.getenv("OPENAI_API_BASE"), 
    api_key=os.getenv("OPENAI_API_KEY") 
)

model_nothinker = LiteLlm(
    model="openai/Qwen3.6-35B-A3B-FP8",
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)


# ==========================================
# FASE 1: Análisis del Problema
# ==========================================
analyzer_agent = LlmAgent(
    name="Analyzer",
    model=model,
    description="Analiza el prompt del usuario para estructurar el problema Multi-Agente.",
    instruction=(
        "Eres el arquitecto principal de un sistema generador de código BDI en Jason.\n"
        "Analiza la petición del usuario y genera un resumen ESTRICTO con 4 puntos:\n"
        "1. OBJETIVO: ¿Qué debe hacer el MAS?\n"
        "2. AGENTES: Lista los agentes que intervienen y sus roles.\n"
        "3. CONCEPTOS TÉCNICOS BDI: Traduce lo que pide el usuario a 3 conceptos técnicos puros de Jason/AgentSpeak en inglés (ej. 'KQML message passing', 'tell', 'recursion', 'bidding').\n"
        "4. PROYECTO: Nombre del proyecto (una sola palabra, en minúsculas)."
    ),
    output_key="analysis_result" 
)


# ==========================================
# FASE 2: Los Investigadores (Paralelo)
# ==========================================
rag_researcher = LlmAgent(
    name="RAG_Researcher",
    model=model,
    description="Busca teoría BDI y sintaxis literal en los PDF locales.",
    instruction=(
        "Usa el análisis previo: {analysis_result}\n"
        "Busca CÓMO se implementan los conceptos técnicos detectados usando `search_local_docs(query)`.\n\n"
        "PROTOCOLO: Máximo 3 llamadas, una por concepto técnico. "
        "Contador interno: [1/3], [2/3], [3/3]. "
        "Al llegar a [3/3] tu turno termina inmediatamente.\n\n"
        "FORMATO DE SALIDA OBLIGATORIO:\n"
        "FRAGMENTOS:\n"
        "[fragmentos literales encontrados, o 'Sin documentación local para [concepto]']\n\n"
        "PROHIBICIONES:\n"
        "- Nunca expliques la sintaxis con tus palabras.\n"
        "- Nunca inventes fragmentos si no los encuentras.\n"
        "- Nunca hagas una cuarta llamada.\n"
    ),
    tools=[rag.search_local_docs],
    output_key="theory_context"
)

github_researcher = LlmAgent(
    name="GitHub_Researcher",
    model=model,
    description="Explora repositorios para extraer ejemplos reales de código Jason.",
    instruction=(
        "Usa el análisis previo: {analysis_result}\n"
        "Tu misión es buscar código real usando `search_github_examples(path)`.\n\n"
        "PROTOCOLO DE EXPLORACIÓN (MÁXIMO 3 LLAMADAS EN TOTAL, SIN EXCEPCIÓN):\n"
        "- Llamada 1 (obligatoria): `path=\"\"` para listar la raíz.\n"
        "- Llamada 2 (obligatoria): elige UNA carpeta relevante y léela.\n"
        "- Llamada 3 (obligatoria): elige UN archivo .asl o .mas2j dentro de esa carpeta y léelo.\n"
        "Tras la tercera llamada, tu turno TERMINA INMEDIATAMENTE. "
        "Aunque no hayas encontrado el ejemplo perfecto. Aunque queden carpetas por explorar.\n\n"
        "CONTADOR INTERNO: Lleva la cuenta en tu cabeza: [1/3], [2/3], [3/3]. "
        "Al llegar a [3/3], la siguiente acción es siempre tu respuesta final, nunca otra llamada.\n\n"
        "FORMATO DE SALIDA OBLIGATORIO:\n"
        "ARCHIVO: [ruta/del/archivo.asl o 'ninguno']\n"
        "CODIGO:\n"
        "[pega aquí el contenido literal del archivo, o 'No se encontró código relevante']\n\n"
        "PROHIBICIONES:\n"
        "- Nunca escribas explicaciones, opiniones ni resúmenes fuera del formato.\n"
        "- Nunca inventes código ni completes fragmentos incompletos.\n"
        "- Nunca hagas una cuarta llamada a la herramienta."
    ),
    tools=[search_github_examples],
    output_key="github_context"
)

knowledge_gatherer = ParallelAgent(
    name="KnowledgeGatherer",
    sub_agents=[rag_researcher, github_researcher]
)


# ==========================================
# FASE 3: Bucle de Programación y Testeo (Loop)
# ==========================================
coder_agent = LlmAgent(
    name="JasonCoder",
    model=model,
    description="Genera código Jason/ASL sintácticamente válido en formato JSON puro.",
    instruction=(
        "Eres un generador de código Jason/ASL. Tu tarea es producir código correcto basado en:\n"
        "- Análisis: {analysis_result}\n"
        "- Teoría: {theory_context}\n"
        "- Ejemplos: {github_context}\n"
        "- Feedback previo: REVISA EL HISTORIAL: Si el CodeTester ha dejado feedback, "
        "lee su CONSOLA_RAW y corrige el código en consecuencia.\n\n"

        "SALIDA: Devuelve ÚNICAMENTE un objeto JSON válido con dos claves: "
        "'mas2j_code' (string) y 'agents_dict' (objeto). Sin markdown, sin explicaciones.\n\n"

        "REGLAS JASON/ASL CRÍTICAS:\n"
        "1. Identificadores (agentes): minúscula (ej. ping, pong).\n"
        "2. Variables: MAYÚSCULA.\n"
        "3. Creencias y Objetivos Iniciales: Terminan en punto (.).\n"
        "4. EL ARRANQUE (VITAL): El agente que inicia la interacción DEBE tener el objetivo inicial "
        "`!start.` en la línea 1 de su archivo .asl, totalmente suelto. "
        "Debajo, creas el plan `+!start <- ...`. "
        "Si no pones el `!start.` suelto, la consola se quedará en blanco.\n"
        "5. PUNTUACIÓN EN PLANES (Crítico):\n"
        "   - Formato: +evento : contexto <- acción1; acción2; acción3.\n"
        "   - Las acciones se separan con punto-coma (;).\n"
        "   - SOLO la última acción del plan termina con punto (.).\n"
        "6. Mensajes entre agentes:\n"
        "   - Envío: .send(receptor, tell, contenido).\n"
        "   - Recepción: +contenido : true <- acciones.\n"
        "   - PROHIBIDO usar sintaxis inventada como `+m: msg(X)`. Usa directamente el contenido.\n"
        "7. Acciones válidas: SOLO .send(), .print().\n"
        "   - PROHIBIDO: .println(), .stopMAS, .stopGUI, +!bucle.\n"
        "8. Contadores: `N1 = N + 1; -+creencia(N1).` Prohibido: `-+creencia(N+1)`.\n"
        "9. Recursión: usar `!bucle` en el cuerpo. Prohibido: `+!bucle`.\n"
        "10. FORMATO MAS2J OBLIGATORIO:\n"
        "    MAS nombre {\n"
        "        infrastructure: Local\n"
        "        agents:\n"
        "            agente1;\n"
        "            agente2;\n"
        "    }\n"
        "11. FORMATO JSON: Usa `\\n` para saltos de línea dentro del código ASL.\n\n"

        "ESTRATEGIA DE CORRECCIÓN:\n"
        "- TIMEOUT_OK sin STDERR limpio: revisa la lógica.\n"
        "- TIMEOUT_OK con warnings en STDERR: corrígelos.\n"
        "- EXITO: devuelve el JSON anterior sin cambios.\n"
        "- LIMITE_SUPERADO: devuelve tu mejor versión.\n"
        "- Cualquier ERROR: lee CONSOLA_RAW y corrige.\n"
    ),
    output_key="generated_code"
)

tester_agent = LlmAgent(
    name="CodeTester",
    model=model_nothinker,
    description="Ejecuta el código del JasonCoder una única vez y devuelve el log de consola sin modificarlo.",
    instruction=(
        "Eres un ejecutor de consola. Sigue estos pasos sin excepción:\n\n"
        "1. Extrae 'mas2j_code' y 'agents_dict' del JSON en {generated_code}.\n"
        "2. Llama a `test_mas_code` UNA SOLA VEZ con esos valores exactos.\n"
        "3. Devuelve ÚNICAMENTE esto:\n\n"
        "CONSOLA_RAW:\n"
        "[string devuelto por la tool]\n\n"
        "ESTADO: [LIMITE_SUPERADO | TIMEOUT_OK | ERROR | EXITO]\n\n"
        "Tabla de estado:\n"
        "  'ERROR: Límite' → LIMITE_SUPERADO\n"
        "  'AVISO: Tiempo' → TIMEOUT_OK\n"
        "  'ERROR'         → ERROR\n"
        "  otro            → EXITO\n"
    ),
    tools=[test_mas_code],
    output_key="test_feedback"
)

refinement_loop = LoopAgent(
    name="RefinementLoop",
    sub_agents=[coder_agent, tester_agent],
    max_iterations=5,
)


# ==========================================
# FASE 4: Guardado Final
# ==========================================
saver_agent = LlmAgent(
    name="Saver",
    model=model_nothinker,
    description="Agente encargado de guardar el código final.",
    instruction=(
        "Tu tarea es guardar el código.\n"
        "1. Usa la función 'save_mas_code(mas_name, mas2j_code, agents_dict)' para guardar.\n"
        "2. El 'mas_name' es el nombre del proyecto extraído en {analysis_result}.\n"
        "3. El 'mas2j_code' y 'agents_dict' son el resultado final en {generated_code}.\n"
        "4. Devuelve el mensaje de éxito o error de la función.\n"
    ),
    tools=[save_mas_code]
)


# ==========================================
# ORQUESTADOR FINAL
# ==========================================
bdi_developer_mas = SequentialAgent(
    name="BDI_Developer_MAS",
    sub_agents=[
        analyzer_agent, 
        knowledge_gatherer, 
        refinement_loop,
        saver_agent 
    ],
)


# ==========================================
# EJECUCIÓN
# ==========================================
async def test_mas():
    print("🚀 Iniciando el MetaMAS BDI Developer...\n")

    # test_prompt = (
    #     "Crea un sistema multi-agente llamado pingpong. "
    #     "Debe tener dos agentes haciendo un ping: un agente envía un mensaje 'Ping' "
    #     "y el otro le contesta con un 'Pong'. Ambos imprimen el mensaje recibido."
    # )
    
    # test_prompt = (
    #     "Crea un sistema multi-agente llamado pingpong2. "
    #     "Debe tener dos agentes haciendo un ping: un agente envía un mensaje 'Ping' "
    #     "y el otro le contesta con un 'Pong'. Ambos imprimen el mensaje recibido. "
    #     "Este intercambio de mensajes debe repetirse exactamente 3 veces y luego "
    #     "ambos agentes deben detenerse."
    # )

    test_prompt = (
    "Crea un sistema multi-agente llamado fibonacci. "
    "Debe tener un único agente llamado fib que calcule e imprima "
    "la serie de Fibonacci hasta el valor 100. "
    "El agente debe imprimir cada número de la serie uno a uno."
    )
    
    print(f"Prompt de prueba:\n{test_prompt}\n")
    print("Ejecutando agentes...\n")
    
    try:
        runner = InMemoryRunner(agent=bdi_developer_mas)
        await runner.run_debug(test_prompt, verbose=True)
        print("\n=== TEST FINALIZADO ===")
    except MASCompletedException as e:
        print(f"\n✅ {e}")
        print("\n=== TEST FINALIZADO ===")
    except Exception as e:
        print(f"\nError fatal en el Runner: {e}")


if __name__ == "__main__":
    asyncio.run(test_mas())