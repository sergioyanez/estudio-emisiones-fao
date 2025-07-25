# Estudio de Emisiones FAO

Este proyecto analiza y visualiza datos de emisiones globales utilizando **Python** y **Streamlit**.  
Puedes ejecutar la aplicación de forma local o mediante **Docker**.

---

## 🚀 **Cómo clonar el repositorio**

```bash
git clone https://github.com/sergioyanez/estudio-emisiones-fao.git
cd estudio-emisiones-fao

💻 Ejecución con Python
1. Crear un entorno virtual (opcional, recomendado)
  python3 -m venv venv
  source venv/bin/activate    # En Linux/Mac
  venv\Scripts\activate       # En Windows (PowerShell)

2. Instalar dependencias
  pip install -r requirements.txt

3. Ejecutar la app
  streamlit run app.py
  Luego abre el navegador en http://localhost:8501.

🐳 Ejecución con Docker
Este proyecto incluye un Dockerfile y docker-compose.yml para facilitar su despliegue.

1. Construir y levantar el contenedor
  docker-compose up --build
2. Acceder a la app
  Abre tu navegador en http://localhost:8501.


📂 Estructura principal del proyecto
bash
Copiar
Editar
.
├── app.py                # Código principal de la app
├── Dockerfile            # Imagen Docker
├── docker-compose.yml    # Configuración Docker
├── requirements.txt      # Dependencias del proyecto
└── .gitignore            # Archivos y carpetas ignoradas por Git


🛠 Requisitos
Python 3.9+ (si usas ejecución local)

Docker y Docker Compose (si prefieres usar contenedores)

Streamlit (instalado automáticamente con requirements.txt)

👤 Autor
Sergio Yáñez
GitHub
