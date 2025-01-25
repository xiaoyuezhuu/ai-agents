from crewai.tools import BaseTool
from typing import Type, Union, List, Dict
from pydantic import BaseModel, Field
import os
import zipfile
import shutil
import json
from typing import Type, Any, Literal, Optional
import subprocess
import requests

SWE_OUTPUT_DIR = "outputs/mvp"
QA_OUTPUT_DIR = "outputs/qa_test"

# class MyCustomToolInput(BaseModel):
#     """Input schema for MyCustomTool."""
#     argument: str = Field(..., description="Description of the argument.")


# class MyCustomTool(BaseTool):
#     name: str = "Name of my tool"
#     description: str = (
#         "Clear description for what this tool is useful for, your agent will need this information to use it."
#     )
#     args_schema: Type[BaseModel] = MyCustomToolInput

#     def _run(self, argument: str) -> str:
#         # Implementation goes here
#         return "this is an example of a tool output, ignore it and move along."


# ✅ Define a rel axed input schema
class SaveCodeFileInput(BaseModel):
    filename: str = Field(..., description="The name of the file (e.g., 'index.html').")
    content: Any = Field("", description="The content to be written in the file.")


### 📂 Custom Tool: Save Code Files
class SaveCodeFileToolSWE(BaseTool):
    name: str = "Save Code File"
    description: str = (
        "Saves generated code into a file within the MVP output directory."
    )
    args_schema: Type[BaseModel] = SaveCodeFileInput

    def _run(self, filename: str, content: str) -> str:
        # ✅ Ensure input is parsed correctly
        if isinstance(filename, str) and filename.startswith("{"):
            try:
                data = json.loads(filename)  # Parse JSON string
                filename = data.get("filename")
                content = data.get("content", "")
            except json.JSONDecodeError:
                return "❌ Error: Input is not a valid JSON dictionary."

        # Ensure all parent directories exist before writing the file
        filepath = os.path.join(SWE_OUTPUT_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Creates missing folders

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

        return f"✅ File saved: {filepath}"


class ManageCodeFileInput(BaseModel):
    action: Literal["move", "delete"] = Field(
        ..., description="Action to perform: 'move' or 'delete'."
    )
    source_path: str = Field(..., description="Full path of the source file.")
    destination_path: Optional[str] = Field(
        None, description="Destination path (only required for moving)."
    )


class ManageCodeFileTool(BaseTool):
    name: str = "Manage Code File"
    description: str = (
        "Allows moving or deleting files within the MVP output directory."
    )
    args_schema: Type[BaseModel] = ManageCodeFileInput

    def _run(
        self, action: str, source_path: str, destination_path: Optional[str] = None
    ) -> str:
        # ✅ Ensure paths are within the allowed directory
        source_full_path = os.path.join(SWE_OUTPUT_DIR, source_path)

        if not os.path.exists(source_full_path):
            return f"❌ Error: File '{source_full_path}' not found!"

        if action == "delete":
            try:
                os.remove(source_full_path)
                return f"🗑️ File deleted: {source_full_path}"
            except Exception as e:
                return f"❌ Error deleting file: {str(e)}"

        elif action == "move":
            if not destination_path:
                return "❌ Error: 'destination_path' is required for moving files."

            destination_full_path = os.path.join(SWE_OUTPUT_DIR, destination_path)
            os.makedirs(
                os.path.dirname(destination_full_path), exist_ok=True
            )  # Ensure destination directory exists

            try:
                shutil.move(source_full_path, destination_full_path)
                return f"📂 File moved: {source_full_path} ➝ {destination_full_path}"
            except Exception as e:
                return f"❌ Error moving file: {str(e)}"

        return "❌ Invalid action. Use 'move' or 'delete'."


class SaveCodeFileToolQA(BaseTool):
    name: str = "Save Code File"
    description: str = (
        "Saves generated code into a file within the MVP output directory."
    )
    args_schema: Type[BaseModel] = SaveCodeFileInput

    def _run(self, filename: str, content: str) -> str:
        # ✅ Ensure input is parsed correctly
        if isinstance(filename, str) and filename.startswith("{"):
            try:
                data = json.loads(filename)  # Parse JSON string
                filename = data.get("filename")
                content = data.get("content", "")
            except json.JSONDecodeError:
                return "❌ Error: Input is not a valid JSON dictionary."

        # Ensure all parent directories exist before writing the file
        filepath = os.path.join(QA_OUTPUT_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Creates missing folders

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

        return f"✅ File saved: {filepath}"


# ✅ Define an empty input schema
class EmptyInputSchema(BaseModel):
    """Empty schema because this tool requires no inputs."""

    pass


### 🗜️ Custom Tool: Zip MVP Code
class ZipMVPCodeTool(BaseTool):
    name: str = "Zip MVP Code"
    description: str = (
        "Compresses all files in the MVP output directory into a single .zip file."
    )
    args_schema: Type[BaseModel] = EmptyInputSchema

    def _run(self) -> str:
        zip_filename = "mvp_code.zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(SWE_OUTPUT_DIR):
                for file in files:
                    zipf.write(os.path.join(root, file), file)

        return f"Code compressed into {zip_filename}"


class ReadFileInputSchema(BaseModel):
    path: str = Field(..., description="Path to a file or directory.")


class ReadFileTool(BaseTool):
    name: str = "Read Files"
    description: str = (
        "Reads a specified file or all files within a directory. "
        "Always run this first to understand the project requirements."
    )
    args_schema: Type[BaseModel] = ReadFileInputSchema

    def _run(self, path: str) -> Union[str, Dict[str, str]]:
        if not os.path.exists(path):
            return f"❌ Error: '{path}' not found!"

        if os.path.isfile(path):
            return self._read_file(path)

        elif os.path.isdir(path):
            return self._read_directory(path)

        return f"❌ Error: '{path}' is neither a file nor a directory!"

    def _read_file(self, filepath: str) -> str:
        """Reads a single file's content."""
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
            return f"✅ File content from {filepath}:\n\n{content}"
        except Exception as e:
            return f"❌ Error reading file '{filepath}': {str(e)}"

    def _read_directory(self, dirpath: str) -> Dict[str, str]:
        """Reads all files in a directory recursively and returns a structured dictionary."""
        file_contents = {}

        for root, _, files in os.walk(dirpath):
            for file in files:
                file_path = os.path.join(root, file)
                file_contents[file_path] = self._read_file(file_path)

        return file_contents


class RunCommandInput(BaseModel):
    """Input schema for RunCommandTool."""

    command: str


class RunCommandTool(BaseTool):
    name: str = "Run Command"
    description: str = "Executes a shell command and returns the output."
    args_schema: Type[BaseModel] = RunCommandInput

    def _run(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,  # ⏳ Set a timeout to prevent hanging
                input="\n",  # Prevents interactive prompts from blocking execution
            )
            return (
                result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
            )
        except subprocess.TimeoutExpired:
            return f"❌ Timeout: The command '{command}' took too long!"
        except Exception as e:
            return f"❌ Error executing command: {str(e)}"


class APITestInput(BaseModel):
    """Input schema for APITestTool."""

    url: str
    method: str = "GET"
    data: dict = None


class APITestTool(BaseTool):
    name: str = "API Test"
    description: str = (
        "Sends an HTTP request to the given API endpoint and returns the response."
    )
    args_schema: Type[BaseModel] = APITestInput

    def _run(self, url: str, method: str = "GET", data: dict = None) -> str:
        try:
            response = requests.request(method, url, json=data)
            return f"Status: {response.status_code}\nResponse: {response.text}"
        except Exception as e:
            return f"Error testing API endpoint: {str(e)}"
