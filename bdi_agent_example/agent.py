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
        if best_error_count == float('inf'):
            best_mas_state = {"mas2j": mas2j_code, "agents": agents_dict}
        return f"AVISO: Tiempo límite (15s). Normal si arranca GUI."
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
def terminar():
    sys.exit()


# ==========================================
# INICIALIZACIÓN DEL MODELO
# ==========================================
model = LiteLlm(
    model="openai/Qwen3.6-35B-A3B-FP8",
    api_base=os.getenv("OPENAI_API_BASE"), 
    api_key=os.getenv("OPENAI_API_KEY") 
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
    description="Busca teoría BDI y sintaxis en los PDF locales.",
    instruction=(
        "Usa el análisis previo: {analysis_result}\n"
        "ESTÁS OBLIGADO a llamar a la herramienta 'search_local_docs(query)'.\n"
        "Usa como 'query' los conceptos técnicos para extraer la teoría sobre cómo se programan esas acciones. Resume la sintaxis obligatoria."
        "ESTÁ ESTRÍCTAMENTE PROHIBIDO que escribas código .asl o .mas2j para resolver el problema del usuario."
        "Devuelve ÚNICAMENTE un resumen de la sintaxis teórica que has encontrado."
    ),
    tools=[rag.search_local_docs],
    output_key="theory_context"
)

github_researcher = LlmAgent(
    name="GitHub_Researcher",
    model=model,
    description="Busca ejemplos de arquitectura y sintaxis BDI en GitHub.",
    instruction=(
        "Usa el análisis previo: {analysis_result}\n"
        "Tu misión es encontrar ejemplos de código real para evitar errores de sintaxis.\n"
        "PASO 1: Llama a search_github_examples(path=\"\") para ver las carpetas.\n"
        "PASO 2: Entra en una carpeta relevante (ej. si ves 'ping', llama a search_github_examples(path=\"ping\") para listar su interior).\n"
        "PASO 3: ESTÁS OBLIGADO a leer el contenido de un archivo con extensión '.mas2j' para ver su estructura exacta, llamando a search_github_examples(path=\"carpeta_elegida/archivo.mas2j\").\n"
        "Devuelve un resumen con el contenido exacto del .mas2j encontrado y los comandos clave de los .asl."
        "ESTÁ ESTRÍCTAMENTE PROHIBIDO que escribas código .asl o .mas2j para resolver el problema del usuario."
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
    description="Programa el MAS respetando estrictamente la sintaxis de Jason.",
    instruction=(
    "Basado en el análisis: {analysis_result}, Teoría: {theory_context}, y Ejemplos: {github_context}.\n"
    "Ten en cuenta el feedback de los tests anteriores.\n\n"

    "REGLAS CRÍTICAS PARA JASON/ASL:\n"
    "1. Devuelve SOLO JSON válido.\n"
    "2. NO uses markdown.\n"
    "3. NO uses ```.\n"
    "4. Variables en Mayúscula.\n"
    "5. Átomos y predicados en minúscula.\n"
    "6. Todo plan, creencia y objetivo termina con punto (.).\n"
    "7. Usa sintaxis válida de AgentSpeak(Jason).\n"
    "8. Los planes deben usar el formato:\n"
    "   +trigger : context <- body.\n"
    "9. No inventes acciones internas inexistentes.\n"
    "10. Las acciones internas deben empezar por punto.\n"
    "11. No uses sintaxis de otros lenguajes.\n"
    "12. Los mensajes usan .send.\n"
    "13. No dejes planes vacíos.\n"
    "14. No uses comentarios.\n"
    "15. El .mas2j debe ser sintácticamente válido.\n"
    "16. Incluye todos los agentes declarados en mas2j.\n"
    "17. El envío de mensajes se realiza con .send(receptor, tell, contenido). El receptor reacciona directamente con un plan cuyo disparador sea +contenido, nunca capturando estructuras abstractas como +msg()."
    "18. Cada archivo .asl debe contener únicamente código ASL.\n\n"

    "Formato EXACTO:\n"
    "{\n"
    '  "mas2j": "...",\n'
    '  "agents": {\n'
    '    "agent1.asl": "...",\n'
    '    "agent2.asl": "..."\n'
    "  }\n"
    "}\n"
"""
...
REGLA DE ORO: Si en el historial del feedback ya aparece la palabra 'ÉXITO',
tu ÚNICA tarea es devolver el mismo código JSON generado anteriormente, sin cambios. 
NO intentes optimizarlo ni cambiarlo, simplemente vuelve a imprimir el JSON ya existente.
"""
),
    output_key="generated_code"
)

tester_agent = LlmAgent(
    name="CodeTester",
    model=model,
    description="Prueba el código generado y decide si iterar o salir del bucle.",
    instruction=("Tienes este código recién generado: {generated_code}\n"
        "PASO 1: Llama a 'test_mas_code(mas2j_code, agents_dict)'. Asegúrate de que las claves en 'agents_dict' sean exactamente los nombres definidos en el .mas2j .\n"
        "PASO 2: Analiza el resultado:\n"
        "   - Si el test es exitoso (timeout o éxito), escribe ÚNICAMENTE: 'ÉXITO: [Resumen corto]'.\n"
        "   - Si hay errores, escribe un feedback detallado para que el JasonCoder lo corrija.\n"
        "   - Si se supera el límite de intentos, escribe: 'ERROR: Límite de 5 intentos superado'."
        "REGLA DE ORO: 1. Si el código generado ya ha sido marcado como 'ÉXITO' en una iteración previa, devuelve inmediatamente: 'ÉXITO: Código ya verificado, saltando testeo.' y NO llames a la herramienta 'test_mas_code'."
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
    model=model,
    description="Agente encargado de guardar el código final una vez que el bucle de refinamiento ha concluido exitosamente.",
    instruction=(
        "Una vez que el bucle de refinamiento ha concluido exitosamente, tu tarea es guardar el código final.\n"
        "Usa la función 'save_mas_code(mas_name, mas2j_code, agents_dict)' para guardar el proyecto.\n"
        "El 'mas_name' debe ser el nombre del proyecto extraído en el {analysis_result}.\n"
        "El 'mas2j_code' y 'agents_dict' deben ser el resultado final del bucle de refinamiento ({generated_code}).\n"
        "Devuelve un mensaje de éxito o error basado en la respuesta de la función de guardado."
        "En caso de éxito deberás llamar a 'terminar()'"
    ),
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
# EJECUCIÓN DEL TEST
# ==========================================

async def test_mas():
    print("🚀 Iniciando el MetaMAS BDI Developer...\n")
    
    test_prompt = (
        "Crea un sistema multi-agente llamado pingpong. "
        "Debe tener dos agentes haciendo un ping: un agente envía un mensaje 'Ping' "
        "y el otro le contesta con un 'Pong'. Ambos imprimen el mensaje recibido."
    )
    
    print(f"📝 Prompt de prueba:\n{test_prompt}\n")
    print("⏳ Ejecutando agentes (arrancando el motor InMemoryRunner)...\n")
    
    try:
        # En ADK, el Runner es el que gestiona la sesión y lanza la ejecución de los agentes
        runner = InMemoryRunner(agent=bdi_developer_mas)
        
        # El método run_debug es ideal aquí porque nos imprimirá por consola 
        # todo el flujo mental, llamadas a RAG y testeos de los agentes.
        await runner.run_debug(test_prompt, verbose=True)
        print("\n=== ✨ TEST FINALIZADO ✨ ===")
        print("Si el testeo ha ido bien, revisa la carpeta 'output/pingpong' para ver los archivos.")
    
        
    except Exception as e:
        print(f"\n❌ Error fatal en el Runner: {e}")
    finally:
        sys.exit()
if __name__ == "__main__":
    asyncio.run(test_mas())