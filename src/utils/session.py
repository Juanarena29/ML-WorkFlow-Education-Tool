"""
Gestion de estado y sesion del proyecto.
Este modulo centraliza todo el estado persistente de la app usando st.session_state
Define la clase MLProject que representa un proyecto completo de Machine Learning y 
permite controlar el flujo paso a paso
carga -> tipos -> limpieza -> entrenamiento -> resultados
"""

from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import streamlit as st


@dataclass
class MLProject:
    """
    Representa el estado completo de un proyecto de ML
    Esta clase actua como una "unica fuente de verdad" del proyecto
    dataset, config, modelos entrenados, metricas y metadatos.
    vive dentro de st.session_state y se va completando a medida que 
    el usuario avanza por las paginas de la app.
    """
    # datos
    df_original: Optional[pd.DataFrame] = None
    df_limpio: Optional[pd.DataFrame] = None
    # modo de ejecucion
    ui_mode: Literal["learn", "tool"] = "tool"
    home_completed: bool = False
    # modo runtime demo o full
    runtime_mode: Literal["demo", "full"] = "full"
    # tipos de columnas y target
    column_types: Dict[str, str] = field(default_factory=dict)
    # nombre del target
    target_column: Optional[str] = None
    # regression, classification, etc
    problem_type: Optional[str] = None
    # configuracion de limpieza
    cleaning_config: Dict[str, Any] = field(default_factory=dict)
    """ field(default_factory=dict) crea un dict vacio {} por defecto en ese campo"""
    # modelos y pipeline
    trained_models: Dict[str, Any] = field(default_factory=dict)
    preprocessing_pipeline: Optional[Any] = None
    # config de entrenamiento
    """este field, congfigura el campo para que cada instancia de MLProject
    reciba un diccionario unico generado por lambda"""
    train_test_split_config: Dict[str, Any] = field(default_factory=lambda: {
        'test_size': 0.2,
        'random_state': 22,
        'stratify': True  # solo para clasifiacion
    })
    # Resultados
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    target_label_encoder: Optional[Any] = None

    # Metadatos
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        'created_at': datetime.now().isoformat(),
        'project_name': 'untitled project',
        'last_modified': datetime.now().isoformat()
    })

    # validaciones
    validation_errors: List[str] = field(default_factory=list)

    def update_metadata(self) -> None:
        """
        actualiza la metadata de la ultima modificacion del proyecto
        se llama automaticamente despues de operaciones importantes
        (por ejemplo cada etapa del proyecto)
        """
        self.metadata['last_modified'] = datetime.now().isoformat()

    def is_step_completed(self, step: str) -> bool:
        """
        indica si un paso del flujo de la app ya fue completado 
        para poder controlar la navegacion entre paginas y evitar 
        se accedan a etapas sin completar otra necesaria (ejemplo
        ir a entrenar modelo sin cargar datos)
        arg step: nombre del paso (load, types, cleaning, eda, training)
        return true si el paso esta completo y false si no.
        """
        step_checks = {
            'home': self.home_completed is True,
            'load': self.df_original is not None,
            'types': (
                self.df_original is not None and
                len(self.column_types) > 0 and
                self.target_column is not None and
                self.problem_type is not None
            ),
            'cleaning': self.df_limpio is not None,
            'eda': self.df_limpio is not None,
            'training': len(self.trained_models) > 0
        }

        return step_checks.get(step, False)

    def get_feature_columns(self) -> List[str]:
        """
        Devuelve las columnas que se usarán como features para el modelo

        Excluye:
        - la columna target
        - columnas marcadas como 'id' o 'text'
        retorna: Lista de nombres de columnas features.
        """
        # guard clause: devuelve lista vacia si column_types esta vacio o target_column no esta definido
        if not self.column_types or not self.target_column:
            return []
        # retorna una lista de nombres de las columnas que se encuentren en el column_types
        # y que cumpla que -> 'col' no es la columna target y 'col_type' no es ni text ni ID
        return [
            # items() devuelve pares (clave,valor)
            col for col, col_type in self.column_types.items()
            if col != self.target_column and col_type not in ['id', 'text']
        ]

    def get_numeric_features(self) -> List[str]:
        """
        Obtiene las features numéricas del dataset sin incluir el target

        Útil para:
        - aplicar escalado
        - generar visualizaciones numéricas
        """
        # retorna la lista de nombres de las columnas que se encuentran en el column_types
        # y que cumpla que 'col_type' es numerico y que 'col' no es la columna target
        return [
            col for col, col_type in self.column_types.items()
            if col_type == 'numeric' and col != self.target_column
        ]

    def get_categorical_features(self) -> List[str]:
        """
        Obtiene las features categóricas del dataset sin incluir el target
        Útil para:
        - OneHotEncoder
        - análisis y visualizaciones categóricas
        """
        # retorna la lista de nombres de las columnas que se encuentran en el column_types
        # y que cumpla que 'col_type' es categorica y que 'col' no es la columna target
        return [
            col for col, col_type in self.column_types.items()
            if col_type == 'categorical' and col != self.target_column
        ]

    def validate_for_training(self) -> List[str]:
        """
        Ejecuta todas las validaciones necesarias antes de entrenar modelos.

        No lanza excepciones: devuelve una lista de errores/advertencias
        para mostrarlas en la UI y decidir si se permite continuar.

        Returns:
            Lista de mensajes de error (vacia si está todo correcto).
        """
        errors = []

        if self.df_limpio is None:
            errors.append(
                "No hay dataset limpio. Complete el paso de tratamiento de datos.")
            return errors

        if not self.target_column:
            errors.append("No se ha definido una columna target.")
            return errors

        if self.target_column not in self.df_limpio.columns:
            errors.append(
                f"La columna target '{self.target_column}' no existe en el dataset.")
            return errors

        # Validar NaNs en target
        if self.df_limpio[self.target_column].isna().any():
            nan_count = self.df_limpio[self.target_column].isna().sum()
            errors.append(
                f"El target '{self.target_column}' tiene {nan_count} valores NaN. Debe limpiarse primero.")

        # Validar features
        feature_cols = self.get_feature_columns()
        if len(feature_cols) == 0:
            errors.append(
                "No hay columnas features para entrenar. Debe haber al menos una.")

        # Validaciones específicas de clasificación
        # si el problema es de clasificacion
        # verifico que el target tenga al menos 2 clases (ejemplo: si, no)
        # si tiene demasiadas clases, aviso que probablemente no sea una categoria real
        if self.problem_type == 'classification':
            # n_classes = numero de clases distintas del target
            n_classes = self.df_limpio[self.target_column].nunique()
            if n_classes < 2:
                errors.append(
                    f"Clasificación requiere al menos 2 clases. Target tiene {n_classes}.")
            elif n_classes > 50:
                errors.append(
                    f"Advertencia: Target tiene {n_classes} clases. Revise si es correcto.")

        # Tamaño mínimo del dataset
        if len(self.df_limpio) < 10:
            errors.append(
                f"Dataset muy pequeño ({len(self.df_limpio)} filas). Mínimo recomendado: 10.")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la configuración del proyecto a un diccionario serializable.

        Se utiliza para exportar la configuración del experimento
        (por ejemplo, como JSON). No incluye DataFrames ni modelos.

        Returns:
            Diccionario con la configuración del proyecto.
        """
        return {
            'column_types': self.column_types,
            'target_column': self.target_column,
            'problem_type': self.problem_type,
            'cleaning_config': self.cleaning_config,
            'train_test_split_config': self.train_test_split_config,
            'metrics': self.metrics,
            'metadata': self.metadata
        }


def initialize_session() -> None:
    """
    Inicializa las claves necesarias en st.session_state.

    Se asegura de que exista:
    - project (MLProject)
    - estado de navegación
    - flags de confirmación
    - logs de operaciones
    """
    if 'project' not in st.session_state:
        st.session_state.project = MLProject()

    if 'ui_mode' not in st.session_state:
        st.session_state.ui_mode = None

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Carga Dataset'

    if 'confirmations' not in st.session_state:
        st.session_state.confirmations = {
            'types_confirmed': False,
            'cleaning_confirmed': False,
            'training_started': False
        }
    if 'problem_type_override' not in st.session_state:
        st.session_state.problem_type_override = None
    if 'operation_logs' not in st.session_state:
        st.session_state.operation_logs = []


def reset_project(keep_metadata: bool = False) -> None:
    """
    Reinicia el proyecto actual creando una nueva instancia de MLProject.

    Args:
        keep_metadata: Si es True, conserva el nombre del proyecto anterior.
    """
    old_metadata = None
    if keep_metadata and 'project' in st.session_state:
        old_metadata = st.session_state.project.metadata.copy()

    st.session_state.project = MLProject()
    if old_metadata:
        st.session_state.project.metadata['project_name'] = old_metadata.get(
            'project_name', 'Untitled Project')

    st.session_state.confirmations = {
        'types_confirmed': False,
        'cleaning_confirmed': False,
        'training_started': False
    }
    st.session_state.problem_type_override = None

    st.session_state.operation_logs = []


def add_operation_log(operation: str, details: str, status: str = 'success') -> None:
    """
    Registra una operación realizada por el sistema.

    Útil para auditoría, debugging y transparencia hacia el usuario.

    Args:
        operation: Nombre de la operación (ej: 'load_dataset')
        details: Descripción de lo realizado
        status: Estado ('success', 'warning', 'error')
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation,
        'details': details,
        'status': status
    }

    if 'operation_logs' not in st.session_state:
        st.session_state.operation_logs = []

    st.session_state.operation_logs.append(log_entry)

    if 'project' in st.session_state:
        st.session_state.project.update_metadata()


def get_project() -> MLProject:
    """
    Devuelve el proyecto actual asegurando que la sesión esté inicializada.

    Returns:
    Instancia actual de MLProject.
    """
    initialize_session()
    return st.session_state.project


def get_uimode(self) -> None:
    """retorna el modo de uso (interfaz)"""
    return self["ui_mode"]


def get_rtmode(self) -> None:
    """retorna el modo de uso (runtime)"""
    return self["runtime_mode"]


def check_step_access(required_step: str, show_error: bool = True) -> bool:
    """
    Verifica si el usuario puede acceder a un paso del flujo.

    Se usa al inicio de cada página para evitar que el usuario
    avance sin completar pasos obligatorios.

    Args:
        required_step: Paso previo requerido.
        show_error: Si True, muestra un mensaje de error en la UI.

    Returns:
        True si el acceso está permitido, False si no.
    """
    project = get_project()

    if not project.is_step_completed(required_step):
        if show_error:
            step_names = {
                'home': 'Seleccionar un modo de uso',
                'load': 'Cargar el dataset',
                'types': 'Confirmar tipos y seleccionar target',
                'cleaning': 'Aplicar tratamiento de datos',
                'eda': 'Limpiar los datos',
                'training': 'Entrenar al menos un modelo'
            }
            st.error(
                f" Debe completar el paso previo: {step_names.get(required_step, required_step)}")
        return False

    return True
