"""
Proyecto: Generador de Música Evolutiva
---

Descripción:
Este proyecto implementa un algoritmo evolutivo para generar música a partir de datos MIDI, optimizando la coherencia
y calidad musical. Las secuencias iniciales se seleccionan de un conjunto de datos MIDI y evolucionan a través de
operaciones como cruce, mutación y transformaciones musicales avanzadas. El objetivo es producir composiciones únicas
que respeten principios de teoría musical y sean armónicamente agradables. El resultado final se guarda en formato MIDI.

---
Componentes Clave:
1. Carga de Datos MIDI: Selección y procesamiento de notas exclusivamente de instrumentos de piano, garantizando
   que las notas estén dentro del rango MIDI permitido.
2. Algoritmo Evolutivo:
   - Población Inicial: Se crean secuencias de notas seleccionadas aleatoriamente del conjunto MIDI.
   - Selección: Se eligen las mejores secuencias según su puntuación de aptitud.
   - Cruce: Combina elementos de dos secuencias padres para crear descendencia (punto único, basado en segmentos,
     uniforme).
   - Mutación: Modifica las secuencias con cambios en tono, ritmo o dinámica.
3. Transformaciones Musicales:
   - Aplicación de legato, conducción de voces, acentuación y dinámica.
   - Mejora de transiciones entre frases y ajuste de silencios largos.
4. Función de Aptitud:
   - Evalúa características como tonos únicos, armonía, consistencia rítmica, estructura de frases y suavidad.
   - Penaliza secuencias con incoherencias o notas fuera de rango.
---
Cómo Usar el Proyecto:
1. Preparar el Conjunto de Datos MIDI:
   - Coloque archivos MIDI en el directorio especificado en `DATASET_PATH`.
   - Asegúrese de que los archivos contengan instrumentos de piano.
2. Configurar Parámetros:
   - Ajuste el tamaño de la población, número de generaciones, tasas de mutación y cruce en la sección de configuración
     del código.
3. Ejecutar el Programa:
   - Ejecute el archivo principal. El sistema cargará los datos MIDI, ejecutará el algoritmo evolutivo y generará un
     archivo MIDI con la mejor secuencia en `OUTPUT_MIDI_PATH`.
4. Evaluar Resultados:
   - Revise el archivo generado y ajuste parámetros según sea necesario para mejorar la calidad de las composiciones.
"""

import os
import random
import copy
import logging
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
import math
import pretty_midi
from tqdm import tqdm

# --------------------------- Configuración --------------------------- #

# Ruta al conjunto de datos
DATASET_PATH = '/Users/jaimeuria/ITAM/Databases/GiantMIDI-Piano/surname_checked_midis'

# Parámetros del algoritmo evolutivo
POPULATION_SIZE = 100        # Número de individuos en la población
SELECTION_SIZE = 50         # Número de personas maxima para seleccionar
GENERATIONS = 100           # Número de generaciones para evolucionar
MUTATION_RATE = 0.5         # Probabilidad de mutación para cada individuo
OUTPUT_MIDI_PATH = 'segundo.mid'  #Ruta para guardar el archivo MIDI evolucionado
LOG_FILE_PATH = 'midi_load_log.txt'     #Ruta para guardar el registro de carga del archivo MIDI

# Parametros de mutación
PITCH_SHIFT_RANGE = (-2, 2)   #Semitonos para cambio de tono
TIME_STRETCH_FACTOR = (0.9, 1.1)  #Rango de factor para alargar el tiempo
MIN_PITCH = 45                # Tono MIDI mínimo aceptable (A0)
MAX_PITCH = 85               # Tono MIDI máximo aceptable (C8)
MAX_TIME = 100
MAX_PITCH_JUMP = 12          # Una octava

MAX_SIMULTANEOUS_NOTES = 6  # Notas máximas que se pueden tocar a la vez
VELOCITY_RANGE = (60, 100)   # Rango de velocidad ajustado para una dinámica más clara

# Configuración de ritmo
TEMPO = 100.0                # "beats" por minuto
BEAT_DIVISION = 4            # Número de divisiones por tiempo (por ejemplo, 4 para semicorcheas)

# Constantes derivadas
SECONDS_PER_BEAT = 60.0 / TEMPO
BEAT_DURATION = SECONDS_PER_BEAT / BEAT_DIVISION  # Duración de cada división de tiempo

# Número máximo de archivos MIDI para cargar inicialmente (para pruebas)
MAX_INITIAL_FILES = 200

# Agregue nuevas constantes para una generación de música más fluida
LEGATO_FACTOR = 0.6  # Superposición entre notas consecutivas
MIN_NOTE_DURATION = BEAT_DURATION  # Duración mínima para cualquier nota alineada con la cuadrícula.
MAX_NOTE_DURATION = 2.0  # Duración máxima de cualquier nota.
VELOCITY_SMOOTHING = 0.5  # Factor para suavizar los cambios de velocidad
PHRASE_LENGTH = 5  # Duración típica de una frase musical en tiempos.
IDEAL_PHRASE_LENGTH = 8  # Número ideal de notas por frase
PREFERRED_VELOCITY = 80  # Velocidad media preferida
PREFERRED_VELOCITY_MIN = 60  # Límite inferior para la velocidad media preferida
PREFERRED_VELOCITY_MAX = 100  # Límite superior para la velocidad media preferida

MAX_SCORE = 1000.0  # Puntuación máxima posible

# --------------------------- Estructuras de datos --------------------------- #

# Definir la estructura de una nota musical.
Note = Dict[str, Any]
NoteSequence = List[Note]

# --------------------------- Funciones auxiliares --------------------------- #

# Define major and minor scales for reference
MAJOR_SCALES = {key: [(key + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]] for key in range(12)}
MINOR_SCALES = {key: [(key + interval) % 12 for interval in [0, 2, 3, 5, 7, 8, 10]] for key in range(12)}

def determine_key(sorted_notes):
    # Determina la tonalidad de una secuencia de notas.
    pitch_classes = [note['pitch'] % 12 for note in sorted_notes]
    pitch_counter = Counter(pitch_classes)
    most_common = pitch_counter.most_common()
    if not most_common:
        return None, None

    # Encuentra la tonalidad con mayor coincidencia con escalas mayores y menores
    best_score = 0
    best_key = None
    best_scale_type = None
    for key in range(12):
        major_scale = set(MAJOR_SCALES[key])
        minor_scale = set(MINOR_SCALES[key])
        major_score = sum(1 for pitch in pitch_classes if pitch in major_scale)
        minor_score = sum(1 for pitch in pitch_classes if pitch in minor_scale)
        if major_score > best_score:
            best_score = major_score
            best_key = key
            best_scale_type = 'major'
        if minor_score > best_score:
            best_score = minor_score
            best_key = key
            best_scale_type = 'minor'
    return best_key, best_scale_type, best_score / len(pitch_classes)

def apply_dynamic_balance(note_sequence: NoteSequence, target_mean: float = 80.0) -> NoteSequence:
    """
    Ajusta las velocidades (dinámica) de las notas para mantener un promedio cercano al valor deseado.
    
    Parámetros:
        note_sequence (NoteSequence): Secuencia actual de notas.
        target_mean (float): Promedio deseado de velocidad.

    Retorna:
        NoteSequence: Secuencia ajustada de velocidades.
    """
    if not note_sequence:
        return note_sequence

    current_mean = np.mean([note['velocity'] for note in note_sequence])
    adjustment_factor = target_mean / current_mean if current_mean != 0 else 1.0

    adjusted = copy.deepcopy(note_sequence)
    for note in adjusted:
        new_velocity = int(note['velocity'] * adjustment_factor)
        # Limita las velocidades dentro del rango definido
        note['velocity'] = max(VELOCITY_RANGE[0], min(VELOCITY_RANGE[1], new_velocity))

    return adjusted

def apply_swing(note_sequence: NoteSequence, swing_factor: float = 0.2) -> NoteSequence:
    """
    Aplica un estilo swing a la secuencia de notas al retrasar las notas fuera del pulso principal.

    Parámetros:
        note_sequence (NoteSequence): Secuencia actual de notas.
        swing_factor (float): Fracción del tiempo de un pulso para retrasar (por ejemplo, 0.2 es un retraso del 20%).

    Retorna:
        NoteSequence: Secuencia con swing aplicado.
    """
    swung = copy.deepcopy(note_sequence)
    swung.sort(key=lambda x: x['start'])

    for note in swung:
        # Determina si la nota está en una subdivisión fuera del pulso (por ejemplo, corcheas en 4/4)
        subdivision = note['start'] / BEAT_DURATION
        if abs(subdivision - round(subdivision)) == 0.5:
            # Retrasa la nota en función del swing_factor
            note['start'] += BEAT_DURATION * swing_factor
            note['end'] += BEAT_DURATION * swing_factor
            # Vuelve a cuantizar para mantener alineación
            note['start'] = quantize_time(note['start'])
            note['end'] = quantize_time(note['end'])

    swung.sort(key=lambda x: x['start'])
    return swung

def quantize_time(time: float, division: int = BEAT_DIVISION) -> float:
    """
    Cuantiza un tiempo dado a la división más cercana del pulso.

    Parámetros:
        time (float): Tiempo original en segundos.
        division (int): Número de divisiones por pulso.

    Retorna:
        float: Tiempo cuantizado en segundos.
    """
    return round(time / BEAT_DURATION) * BEAT_DURATION

def validate_note_sequence(note_sequence: NoteSequence) -> NoteSequence:
    """
    Valida y corrige una secuencia de notas para asegurar que todas las notas sean válidas y estén alineadas al grid.

    Parámetros:
        note_sequence (NoteSequence): La secuencia de notas a validar.

    Retorna:
        NoteSequence: Secuencia de notas validada y corregida.
    """
    validated = []
    for note in note_sequence:
        # Asegura que el final de la nota sea posterior al inicio
        if note['end'] <= note['start']:
            note['end'] = note['start'] + MIN_NOTE_DURATION  # Asigna una duración mínima

        # Cuantiza los tiempos de inicio y fin
        note['start'] = quantize_time(note['start'])
        note['end'] = quantize_time(note['end'])

        # Remueve notas fuera del rango aceptable
        if MIN_PITCH <= note['pitch'] <= MAX_PITCH:
            # Asegura que la velocidad esté dentro del rango permitido
            note['velocity'] = max(VELOCITY_RANGE[0], min(127, note['velocity']))
            validated.append(note)
        else:
            # Opcional: registrar o contar la eliminación de notas
            pass  # Las notas fuera del rango simplemente se eliminan

    # Ordena por tiempo de inicio
    validated.sort(key=lambda x: x['start'])

    return validated

def get_scale_pitches(root: int, scale_type: str = 'major') -> set:
    """
    Genera un conjunto de notas basado en la raíz y el tipo de escala.

    Parámetros:
        root (int): Nota raíz en número MIDI.
        scale_type (str): Tipo de escala ('major' o 'minor').

    Retorna:
        set: Conjunto de notas MIDI en la escala.
    """
    if scale_type == 'major':
        intervals = [0, 2, 4, 5, 7, 9, 11]  # Intervalos de la escala mayor
    elif scale_type == 'minor':
        intervals = [0, 2, 3, 5, 7, 8, 10]  # Intervalos de la escala menor natural
    else:
        raise ValueError("Tipo de escala no soportado. Usa 'major' o 'minor'.")

    scale_pitches = set(root + interval for interval in intervals)
    return scale_pitches

def mutate_phrases(note_sequence: NoteSequence, phrase_length_beats: int = 8) -> NoteSequence:
    """
    Realiza mutaciones a frases completas aplicando transformaciones como inversión o retrogradación.

    Parámetros:
        note_sequence (NoteSequence): La secuencia actual de notas.
        phrase_length_beats (int): Número de pulsos que define una frase.

    Retorna:
        NoteSequence: Secuencia de notas con transformaciones aplicadas a nivel de frase.
    """
    if not note_sequence:
        return note_sequence

    mutated = copy.deepcopy(note_sequence)
    mutated.sort(key=lambda x: x['start'])

    # Calcula la duración de cada frase en segundos
    phrase_duration = phrase_length_beats * SECONDS_PER_BEAT
    # Determina el número total de frases en la secuencia
    num_phrases = int(math.ceil(max(note['end'] for note in mutated) / phrase_duration))

    for phrase_idx in range(num_phrases):
        # Extrae las notas dentro de la frase actual
        phrase_notes = [
            note for note in mutated
            if phrase_idx * phrase_duration <= note['start'] < (phrase_idx + 1) * phrase_duration
        ]
        if not phrase_notes:
            continue  # Si no hay notas en la frase, pasa a la siguiente

        # Selecciona una transformación aleatoria para aplicar a la frase
        transformation = random.choice(['invert', 'retrograde', 'transpose_up', 'transpose_down'])
        
        if transformation == 'invert':
            # Invierte las alturas (pitches) alrededor de la nota inicial de la frase
            pivot = phrase_notes[0]['pitch']
            for note in phrase_notes:
                note['pitch'] = pivot - (note['pitch'] - pivot)
        elif transformation == 'retrograde':
            # Reversa el orden temporal de las notas en la frase
            phrase_notes_sorted = sorted(phrase_notes, key=lambda x: x['start'], reverse=True)
            for original, retro in zip(phrase_notes, phrase_notes_sorted):
                original['start'], original['end'] = retro['start'], retro['end']
        elif transformation == 'transpose_up':
            # Transpone toda la frase hacia arriba por un intervalo aleatorio dentro de la escala
            semitones = random.choice([2, 4, 5, 7, 9, 11])  # Intervalos de la escala
            for note in phrase_notes:
                note['pitch'] += semitones
                # Asegura que el pitch esté dentro del rango permitido
                note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))
        elif transformation == 'transpose_down':
            # Transpone toda la frase hacia abajo por un intervalo aleatorio dentro de la escala
            semitones = random.choice([2, 3, 5, 7, 8, 10])  # Intervalos de la escala
            for note in phrase_notes:
                note['pitch'] -= semitones
                # Asegura que el pitch esté dentro del rango permitido
                note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))

    # Ordena nuevamente las notas por tiempo de inicio después de las transformaciones
    mutated.sort(key=lambda x: x['start'])
    return mutated

def apply_voice_leading(note_sequence: NoteSequence) -> NoteSequence:
    """
    Aplica principios de conducción de voces para asegurar transiciones suaves entre las notas.

    Parámetros:
        note_sequence (NoteSequence): La secuencia de notas actual.

    Retorna:
        NoteSequence: Secuencia de notas con conducción de voces aplicada.
    """
    if len(note_sequence) < 2:
        return note_sequence  # Si hay menos de 2 notas, no se puede aplicar conducción de voces

    # Crea una copia de la secuencia para no modificar la original
    led = copy.deepcopy(note_sequence)
    # Ordena las notas por tiempo de inicio
    led.sort(key=lambda x: x['start'])

    for i in range(1, len(led)):
        prev_note = led[i - 1]
        current_note = led[i]

        # Calcula la diferencia de altura (pitch) entre la nota actual y la anterior
        pitch_diff = current_note['pitch'] - prev_note['pitch']

        # Limita los saltos de altura tonal a un máximo de una quinta (7 semitonos)
        if abs(pitch_diff) > 7:
            # Ajusta la altura tonal para reducir el salto
            adjustment = -7 if pitch_diff > 0 else 7
            new_pitch = current_note['pitch'] + adjustment
            # Asegura que la nueva altura tonal esté dentro del rango MIDI permitido
            new_pitch = max(MIN_PITCH, min(MAX_PITCH, new_pitch))
            current_note['pitch'] = new_pitch

    return led

def pitch_shift_scale_conscious(note_sequence: NoteSequence, semitones: int, root: int, scale_type: str = 'major') -> NoteSequence:
    """
    Transpone todas las notas en la secuencia por un número dado de semitonos,
    asegurando que permanezcan dentro de la escala especificada.

    Parámetros:
        note_sequence (NoteSequence): La secuencia de notas actual.
        semitones (int): Número de semitonos para transponer (positivo o negativo).
        root (int): Nota raíz en número MIDI para la generación de la escala.
        scale_type (str): Tipo de escala ('major' o 'minor').

    Retorna:
        NoteSequence: Secuencia transpuesta, ajustada para permanecer en la escala.
    """
    # Obtiene las alturas tonales válidas dentro de la escala
    scale_pitches = get_scale_pitches(root, scale_type)
    shifted = copy.deepcopy(note_sequence)

    for note in shifted:
        # Transpone la nota por el número especificado de semitonos
        note['pitch'] += semitones
        # Asegura que el pitch se mantenga dentro del rango MIDI permitido
        note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))
        
        # Ajusta el pitch para asegurarse de que esté dentro de la escala
        if note['pitch'] not in scale_pitches:
            # Encuentra el pitch más cercano dentro de la escala
            possible_pitches = [p for p in scale_pitches if MIN_PITCH <= p <= MAX_PITCH]
            if possible_pitches:
                closest_pitch = min(possible_pitches, key=lambda p: abs(p - note['pitch']))
                note['pitch'] = closest_pitch
        
        # Cuantiza el tiempo para mantener la alineación rítmica
        note['start'] = quantize_time(note['start'])
        note['end'] = quantize_time(note['end'])
    
    return shifted

def load_single_midi(midi_path: str) -> Optional[pretty_midi.PrettyMIDI]:
    """
    Intenta cargar un archivo MIDI individual.
    Si la carga es exitosa, devuelve un objeto PrettyMIDI; de lo contrario, devuelve None.

    Parámetros:
        midi_path (str): Ruta del archivo MIDI a cargar.

    Retorna:
        Optional[pretty_midi.PrettyMIDI]: El objeto PrettyMIDI si la carga es exitosa; None en caso de error.
    """
    try:
        # Intenta cargar el archivo MIDI usando la biblioteca pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)
        return midi
    except Exception as e:
        # Si ocurre un error, imprime un mensaje y devuelve None
        print(f"Error al cargar {midi_path}: {e}")
        return None

def load_midi_with_path(path: str) -> Tuple[str, Optional[pretty_midi.PrettyMIDI]]:
    """
    Carga un archivo MIDI desde una ruta y devuelve la ruta junto con el objeto PrettyMIDI.

    Parámetros:
        path (str): Ruta completa del archivo MIDI a cargar.

    Retorna:
        Tuple[str, Optional[pretty_midi.PrettyMIDI]]: Una tupla que contiene:
            - La ruta del archivo MIDI.
            - El objeto PrettyMIDI si la carga fue exitosa, o None si ocurrió un error.
    """
    # Utiliza la función load_single_midi para cargar el archivo
    midi = load_single_midi(path)
    # Devuelve la ruta junto con el resultado de la carga
    return (path, midi)


def load_midi_files_parallel(dataset_path: str, max_files: int = None, log_file_path: str = None) -> List[pretty_midi.PrettyMIDI]:
    """
    Carga archivos MIDI desde un directorio utilizando procesamiento paralelo.
    Devuelve una lista de objetos PrettyMIDI cargados exitosamente que contienen instrumentos de piano.

    Parámetros:
        dataset_path (str): Ruta del directorio que contiene los archivos MIDI.
        max_files (int): Número máximo de archivos MIDI a cargar (opcional).
        log_file_path (str): Ruta para guardar un archivo de registro con los nombres de los archivos cargados (opcional).

    Retorna:
        List[pretty_midi.PrettyMIDI]: Lista de objetos PrettyMIDI que se cargaron con éxito y contienen instrumentos de piano.
    """
    midi_files = []
    
    # Obtiene todos los archivos MIDI en el directorio especificado
    all_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.mid', '.midi'))]
    total_files = len(all_files)
    print(f"Total de archivos MIDI encontrados: {total_files}")

    # Si se especifica un límite en max_files, selecciona una muestra aleatoria
    if max_files:
        all_files = random.sample(all_files, min(max_files, len(all_files)))
        total_files = len(all_files)
        print(f"Seleccionados aleatoriamente {total_files} archivos MIDI según el parámetro max_files.")

    # Construye las rutas completas de los archivos MIDI
    all_midi_paths = [os.path.join(dataset_path, f) for f in all_files]

    # Mezcla aleatoriamente las rutas de los archivos MIDI
    random.shuffle(all_midi_paths)
    
    # Usa multiprocessing Pool para cargar archivos en paralelo
    with Pool(processes=cpu_count()) as pool:
        results = tqdm(pool.imap_unordered(load_midi_with_path, all_midi_paths), total=total_files, desc="Cargando archivos MIDI")
        for path, midi in results:
            if midi is not None:
                # Verifica si el archivo MIDI contiene al menos un instrumento de piano
                has_piano = any(
                    instr.program == 0 or 'piano' in instr.name.lower()
                    for instr in midi.instruments if not instr.is_drum
                )
                if has_piano:
                    midi_files.append(midi)
                    # Si se especifica un archivo de registro, guarda el nombre del archivo cargado
                    if log_file_path:
                        filename = os.path.basename(path)
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(f"{filename}\n")
                else:
                    print(f"Archivo omitido {path}: No se encontró instrumento de piano.")
            else:
                # Los errores ya se manejan en load_single_midi
                pass

    print(f"Archivos MIDI cargados exitosamente: {len(midi_files)}")
    return midi_files

def midi_to_note_sequence(instrument: pretty_midi.Instrument) -> NoteSequence:
    """
    Convierte un objeto PrettyMIDI.Instrument a una secuencia de notas (NoteSequence).
    Agrupa las notas que comienzan en tiempos similares para capturar acordes.

    Parámetros:
        instrument (pretty_midi.Instrument): El instrumento PrettyMIDI a convertir.

    Retorna:
        NoteSequence: Una lista de notas con información de tono (pitch), inicio (start),
                      fin (end) y velocidad (velocity).
    """
    # Agrupa las notas por tiempos de inicio similares para identificar acordes
    time_threshold = 0.05  # Las notas que comiencen dentro de 50ms se considerarán parte del mismo acorde
    notes_by_time = {}

    for note in instrument.notes:
        # Redondea el tiempo de inicio al múltiplo más cercano del umbral para agrupar notas
        chord_time = round(note.start / time_threshold) * time_threshold
        if chord_time not in notes_by_time:
            notes_by_time[chord_time] = []
        notes_by_time[chord_time].append({
            'pitch': note.pitch,         # Tono MIDI de la nota
            'start': note.start,         # Tiempo de inicio de la nota
            'end': note.end,             # Tiempo de finalización de la nota
            'velocity': note.velocity    # Velocidad de la nota (dinámica)
        })

    # Aplana las notas agrupadas manteniendo el orden temporal
    notes = []
    for start_time in sorted(notes_by_time.keys()):
        notes.extend(notes_by_time[start_time])

    return notes

def identify_motifs(note_sequence: NoteSequence, motif_length: int = 4) -> List[NoteSequence]:
    """
    Identifica motivos recurrentes dentro de una secuencia de notas.

    Parámetros:
        note_sequence (NoteSequence): La secuencia actual de notas.
        motif_length (int): Número de notas que definen un motivo.

    Retorna:
        List[NoteSequence]: Lista de motivos identificados como sub-secuencias de notas.
    """
    motifs = []
    sequence_length = len(note_sequence)

    # Itera sobre la secuencia para buscar motivos de longitud definida
    for i in range(sequence_length - motif_length + 1):
        # Extrae un candidato a motivo de la secuencia
        candidate = note_sequence[i:i + motif_length]

        # Verifica si el candidato ya fue identificado como motivo
        if candidate in motifs:
            continue  # Si ya está registrado, pasa al siguiente

        # Cuenta cuántas veces se repite el candidato en la secuencia
        count = 1
        for j in range(i + motif_length, sequence_length - motif_length + 1):
            if note_sequence[j:j + motif_length] == candidate:
                count += 1

        # Si el candidato se repite más de una vez, se considera un motivo
        if count > 1:
            motifs.append(candidate)

    return motifs

def mutate_motifs(note_sequence: NoteSequence, motif_length: int = 4) -> NoteSequence:
    """
    Aplica mutaciones a motivos identificados dentro de una secuencia de notas para introducir variaciones.

    Parámetros:
        note_sequence (NoteSequence): La secuencia actual de notas.
        motif_length (int): Número de notas que define un motivo.

    Retorna:
        NoteSequence: Secuencia de notas con variaciones aplicadas a los motivos identificados.
    """
    # Identifica motivos recurrentes en la secuencia
    motifs = identify_motifs(note_sequence, motif_length)
    if not motifs:
        return note_sequence  # Si no hay motivos, devuelve la secuencia original

    # Crea una copia para aplicar las mutaciones
    mutated = copy.deepcopy(note_sequence)

    for motif in motifs:
        # Selecciona un motivo a mutar al azar
        selected_motif = random.choice(motifs)
        # Selecciona una transformación aleatoria para el motivo
        transformation = random.choice(['invert', 'retrograde', 'transpose_up', 'transpose_down'])
        
        # Encuentra todas las apariciones del motivo seleccionado
        indices = [
            i for i in range(len(mutated) - motif_length + 1)
            if mutated[i:i + motif_length] == selected_motif
        ]

        for idx in indices:
            # Obtiene el motivo en la posición actual
            phrase = mutated[idx:idx + motif_length]

            if transformation == 'invert':
                # Invierte las alturas tonales alrededor de la primera nota del motivo
                pivot = phrase[0]['pitch']
                for note in phrase:
                    note['pitch'] = pivot - (note['pitch'] - pivot)

            elif transformation == 'retrograde':
                # Invierte el orden de las notas del motivo (retrogradación)
                reversed_phrase = list(reversed(phrase))
                for original, reversed_note in zip(phrase, reversed_phrase):
                    original['pitch'] = reversed_note['pitch']
                    original['velocity'] = reversed_note['velocity']

            elif transformation == 'transpose_up':
                # Transpone hacia arriba por un intervalo aleatorio dentro de la escala
                semitones = random.choice([2, 4, 5, 7, 9, 11])  # Intervalos de escala mayor
                for note in phrase:
                    note['pitch'] += semitones
                    note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))

            elif transformation == 'transpose_down':
                # Transpone hacia abajo por un intervalo aleatorio dentro de la escala
                semitones = random.choice([2, 3, 5, 7, 8, 10])  # Intervalos de escala menor
                for note in phrase:
                    note['pitch'] -= semitones
                    note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))

    # Ordena las notas por tiempo de inicio después de las mutaciones
    mutated.sort(key=lambda x: x['start'])
    return mutated

def apply_accentuation(note_sequence: NoteSequence, accent_beats: List[int] = [1, 3]) -> NoteSequence:
    """
    Acentúa las notas que caen en los pulsos fuertes especificados.

    Parámetros:
        note_sequence (NoteSequence): La secuencia de notas actual.
        accent_beats (List[int]): Lista de números de pulso (1-indexados) que deben ser acentuados.

    Retorna:
        NoteSequence: La secuencia de notas con acentuaciones aplicadas.
    """
    if not note_sequence:
        return note_sequence  # Devuelve la secuencia original si está vacía

    # Crea una copia de la secuencia para aplicar las modificaciones
    accented = copy.deepcopy(note_sequence)
    accented.sort(key=lambda x: x['start'])  # Ordena las notas por tiempo de inicio

    for note in accented:
        # Determina el número de pulso dentro del compás (1-indexado)
        beat_number = int((note['start'] % SECONDS_PER_BEAT * BEAT_DIVISION) // BEAT_DURATION) + 1
        if beat_number in accent_beats:
            # Incrementa la velocidad para acentuar la nota
            note['velocity'] = min(127, int(note['velocity'] * 1.2))  # Multiplica por 1.2 y asegura que no exceda 127
    
    return accented

def apply_phrase_dynamics(note_sequence: NoteSequence, phrase_length_beats: int = 8) -> NoteSequence:
    """
    Aplica cambios dinámicos a las frases musicales, creando crescendos y diminuendos.

    Parámetros:
        note_sequence (NoteSequence): La secuencia actual de notas.
        phrase_length_beats (int): Número de pulsos por frase.

    Retorna:
        NoteSequence: La secuencia de notas con dinámica aplicada a nivel de frase.
    """
    if not note_sequence:
        return note_sequence  # Si la secuencia está vacía, no hace nada

    # Crea una copia de la secuencia para aplicar las modificaciones
    enhanced = copy.deepcopy(note_sequence)
    enhanced.sort(key=lambda x: x['start'])  # Ordena las notas por tiempo de inicio

    # Calcula los límites de cada frase basándose en la duración de la frase
    phrases = {}
    phrase_duration = phrase_length_beats * SECONDS_PER_BEAT
    for note in enhanced:
        # Identifica a qué frase pertenece la nota actual
        phrase_index = int(note['start'] // phrase_duration)
        if phrase_index not in phrases:
            phrases[phrase_index] = []
        phrases[phrase_index].append(note)

    # Aplica dinámica a cada frase
    for phrase in phrases.values():
        # Decide si la frase tendrá un crescendo o un diminuendo
        trend = random.choice(['crescendo', 'diminuendo'])
        velocities = [note['velocity'] for note in phrase]

        # Calcula la dinámica (incremento o decremento)
        if trend == 'crescendo':
            sorted_velocities = sorted(velocities)
        else:  # diminuendo
            sorted_velocities = sorted(velocities, reverse=True)

        # Aplica la transición dinámica a las notas de la frase
        for i, note in enumerate(phrase):
            note['velocity'] = int(
                np.interp(i, [0, len(phrase) - 1], [sorted_velocities[0], sorted_velocities[-1]])
            )
    
    return enhanced

def smooth_velocities(note_sequence: NoteSequence, window_size: int = 3) -> NoteSequence:
    """
    Suaviza las transiciones de velocidad entre notas consecutivas en la secuencia.

    Parámetros:
        note_sequence (NoteSequence): La secuencia de notas actual.
        window_size (int): Número de notas a incluir en la ventana de suavizado.

    Retorna:
        NoteSequence: La secuencia de notas con velocidades suavizadas.
    """
    if len(note_sequence) < 2:
        return note_sequence  # Si hay menos de 2 notas, no se realiza ningún suavizado

    # Crea una copia de la secuencia para aplicar los cambios
    smoothed = copy.deepcopy(note_sequence)
    smoothed.sort(key=lambda x: x['start'])  # Ordena las notas por tiempo de inicio

    for i in range(len(smoothed)):
        # Construye una ventana de notas alrededor de la nota actual
        window = []
        for j in range(max(0, i - window_size), min(len(smoothed), i + window_size + 1)):
            window.append(smoothed[j]['velocity'])

        # Reemplaza la velocidad de la nota actual por el promedio de la ventana
        smoothed[i]['velocity'] = int(np.mean(window))

    return smoothed

def apply_legato(note_sequence: NoteSequence) -> NoteSequence:
    """
    Crea transiciones más suaves entre notas extendiendo las duraciones para que se superpongan ligeramente.

    Parámetros:
        note_sequence (NoteSequence): La secuencia actual de notas.

    Retorna:
        NoteSequence: Secuencia de notas con legato aplicado.
    """
    if len(note_sequence) < 2:
        return note_sequence  # Si hay menos de 2 notas, no se puede aplicar legato

    # Crea una copia de la secuencia para aplicar las modificaciones
    legato = copy.deepcopy(note_sequence)
    legato.sort(key=lambda x: x['start'])  # Ordena las notas por tiempo de inicio

    for i in range(len(legato) - 1):
        current_note = legato[i]
        next_note = legato[i + 1]

        # Extiende la nota actual para que se superponga ligeramente con la siguiente
        if next_note['start'] > current_note['start']:
            overlap = min(
                LEGATO_FACTOR,  # Define cuánto puede superponerse una nota
                (next_note['start'] - current_note['start']) * 0.5  # Calcula la superposición como la mitad del intervalo
            )
            current_note['end'] = next_note['start'] + overlap  # Extiende la duración de la nota actual

    return legato

def create_phrase_structure(note_sequence: NoteSequence) -> NoteSequence:
    """
    Organiza las notas en frases musicales añadiendo pausas naturales entre ellas.

    Parámetros:
        note_sequence (NoteSequence): La secuencia actual de notas.

    Retorna:
        NoteSequence: Secuencia de notas estructurada en frases con pausas naturales.
    """
    if len(note_sequence) < 4:
        return note_sequence  # Si hay menos de 4 notas, no se modifica la estructura

    # Crea una copia de la secuencia para aplicar las modificaciones
    structured = copy.deepcopy(note_sequence)
    structured.sort(key=lambda x: x['start'])  # Ordena las notas por tiempo de inicio

    # Define la longitud de las frases en pulsos
    phrase_length = PHRASE_LENGTH  # Constante que determina la duración de cada frase en pulsos
    current_phrase_start = structured[0]['start']

    for i in range(len(structured)):
        # Identifica el inicio de una nueva frase
        if structured[i]['start'] - current_phrase_start >= phrase_length:
            # Añade una pequeña pausa al final de la frase anterior
            if i > 0:
                structured[i - 1]['end'] = min(
                    structured[i - 1]['end'],
                    structured[i]['start'] - 0.1  # Reduce el tiempo de finalización de la última nota
                )
            # Actualiza el inicio de la nueva frase
            current_phrase_start = structured[i]['start']

    return structured

def add_dynamic_expression(note_sequence: NoteSequence) -> NoteSequence:
    """
    Añade una expresión dinámica natural a la música, como cambios graduales en la intensidad.

    Parámetros:
        note_sequence (NoteSequence): La secuencia de notas actual.

    Retorna:
        NoteSequence: La secuencia de notas con dinámica expresiva añadida.
    """
    if not note_sequence:
        return note_sequence  # Si la secuencia está vacía, no realiza cambios

    # Crea una copia de la secuencia para aplicar las modificaciones
    expressive = copy.deepcopy(note_sequence)
    expressive.sort(key=lambda x: x['start'])  # Ordena las notas por tiempo de inicio

    # Crea cambios dinámicos basados en una curva natural (en forma de arco)
    phrase_length = PHRASE_LENGTH  # Define la longitud de las frases
    for i in range(len(expressive)):
        # Calcula la posición relativa de la nota dentro de la frase
        position_in_phrase = (expressive[i]['start'] % phrase_length) / phrase_length

        # Aplica un cambio dinámico en forma de arco (sinusoidal)
        dynamic_factor = 1.0 + 0.2 * np.sin(position_in_phrase * np.pi)
        expressive[i]['velocity'] = int(
            min(127, expressive[i]['velocity'] * dynamic_factor)  # Asegura que no exceda el límite MIDI
        )

    return expressive

def note_sequence_to_midi(note_sequence: NoteSequence, tempo: float = TEMPO) -> pretty_midi.PrettyMIDI:
    """
    Convierte una secuencia de notas en un archivo MIDI con expresión musical mejorada.

    Parámetros:
        note_sequence (NoteSequence): La secuencia de notas a convertir.
        tempo (float): Tempo inicial del archivo MIDI en BPM (opcional, valor por defecto definido por TEMPO).

    Retorna:
        pretty_midi.PrettyMIDI: Un objeto PrettyMIDI que representa la secuencia convertida.
    """
    # Aplica mejoras musicales a la secuencia
    enhanced_sequence = note_sequence
    enhanced_sequence = smooth_velocities(enhanced_sequence)  # Suaviza las velocidades
    enhanced_sequence = apply_legato(enhanced_sequence)       # Aplica legato
    enhanced_sequence = create_phrase_structure(enhanced_sequence)  # Crea estructura de frases
    enhanced_sequence = add_dynamic_expression(enhanced_sequence)   # Añade expresión dinámica

    # Crea un nuevo objeto PrettyMIDI con el tempo inicial
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Agrupa las notas por su tiempo de inicio
    notes_by_time = {}
    for note in enhanced_sequence:
        start_time = round(note['start'], 2)  # Redondea para agrupar
        if start_time not in notes_by_time:
            notes_by_time[start_time] = []
        notes_by_time[start_time].append(note)

    # Crea un único instrumento (piano acústico) para el archivo MIDI
    piano = pretty_midi.Instrument(program=0, name='Acoustic Grand Piano')  # Piano acústico

    for start_time, notes in notes_by_time.items():
        # Limita la cantidad de notas simultáneas
        notes = sorted(notes, key=lambda x: x['velocity'], reverse=True)[:MAX_SIMULTANEOUS_NOTES]

        for note in notes:
            midi_note = pretty_midi.Note(
                velocity=int(min(127, max(40, note['velocity']))),  # Asegura que la velocidad esté en rango
                pitch=int(min(127, max(0, note['pitch']))),         # Asegura que el tono esté en rango MIDI
                start=float(note['start']),
                end=float(note['end'])
            )
            piano.notes.append(midi_note)

    # Añade el instrumento al archivo MIDI
    if piano.notes:
        midi.instruments.append(piano)

    return midi

# --------------------------- Operadores de Mutación --------------------------- #
def add_chord(note_sequence: NoteSequence) -> NoteSequence:
    """
    Añade un acorde basado en principios de teoría musical, alineado con patrones rítmicos.
    Solo se añaden acordes armoniosos (triadas y extensiones).

    Parámetros:
        note_sequence (NoteSequence): La secuencia de notas actual.

    Retorna:
        NoteSequence: La secuencia de notas con el acorde añadido.
    """
    if not note_sequence:
        return note_sequence

    # Selecciona una nota aleatoria como raíz, priorizando notas con mayor duración
    valid_notes = [note for note in note_sequence
                   if (note['end'] - note['start']) >= BEAT_DURATION]  # Notas de al menos un pulso
    if not valid_notes:
        return note_sequence

    root_note = random.choice(valid_notes)

    chord_types = [
        # Triadas
        [0, 4, 7],          # Mayor
        [0, 3, 7],          # Menor
        [0, 3, 6],          # Disminuida
        [0, 4, 8],          # Aumentada
        [0, 5, 7],          # Quinta (sin tercera)
        [0, 2, 7],          # Suspendida segunda (sus2)
        [0, 5, 7],          # Suspendida cuarta (sus4)
        [0, 4, 7, 11],      # Séptima mayor
        [0, 3, 7, 10],      # Séptima menor
        [0, 4, 7, 10],      # Séptima dominante

        # Acordes extendidos y alterados
        [0, 4, 7, 11, 14],  # Treceava mayor
        [0, 3, 7, 14],      # Menor add9
        [0, 4, 7, 5],       # Add4
    ]

    # Selecciona un tipo de acorde aleatorio
    intervals = random.choice(chord_types)

    # Añade las notas del acorde con la misma duración que la nota raíz
    new_notes = []
    for interval in intervals[1:]:  # Omite la raíz, ya está incluida
        chord_pitch = root_note['pitch'] + interval
        # Asegura que el pitch esté dentro del rango MIDI permitido
        if MIN_PITCH <= chord_pitch <= MAX_PITCH:
            chord_velocity = int(root_note['velocity'] * 0.8)  # Reduce la intensidad de las notas armónicas
            chord_start = quantize_time(root_note['start'])    # Alinea al grid rítmico
            chord_end = quantize_time(chord_start + (root_note['end'] - root_note['start']))
            chord_note = {
                'pitch': chord_pitch,
                'start': chord_start,
                'end': chord_end,
                'velocity': chord_velocity
            }
            new_notes.append(chord_note)

    note_sequence.extend(new_notes)
    note_sequence.sort(key=lambda x: x['start'])  # Ordena las notas por tiempo de inicio

    return note_sequence

def time_stretch(note_sequence: NoteSequence, factor: float) -> NoteSequence:
    """
    Estira el tiempo de todas las notas por un factor dado, asegurando alineación con el grid rítmico.

    Parámetros:
        note_sequence (NoteSequence): La secuencia actual de notas.
        factor (float): Factor de estiramiento (por ejemplo, 1.1 para alargar duraciones).

    Retorna:
        NoteSequence: La secuencia de notas con los tiempos estirados.
    """
    stretched = copy.deepcopy(note_sequence)
    for note in stretched:
        note['start'] *= factor
        note['end'] *= factor
        # Cuantiza los tiempos para mantener la alineación rítmica
        note['start'] = quantize_time(note['start'])
        note['end'] = quantize_time(note['end'])
    return stretched

def modify_random_note_improved(note_sequence: NoteSequence) -> NoteSequence:
    """
    Modifica un atributo aleatorio de una nota aleatoria en la secuencia,
    asegurando alineación al grid rítmico y coherencia musical.

    Parámetros:
        note_sequence (NoteSequence): La secuencia actual de notas.

    Retorna:
        NoteSequence: La secuencia de notas modificada.
    """
    if not note_sequence:
        return note_sequence

    idx = random.randint(0, len(note_sequence) - 1)  # Selecciona una nota al azar
    note = note_sequence[idx]
    modification = random.choice(['pitch', 'velocity', 'duration'])  # Selecciona un atributo a modificar

    if modification == 'pitch':
        # Modifica el pitch por un intervalo pequeño
        semitones = random.choice([-2, -1, 1, 2])
        note['pitch'] += semitones
        note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))
    elif modification == 'velocity':
        # Ajusta la velocidad dentro de un rango razonable
        change = random.choice([-10, -5, 5, 10])
        note['velocity'] += change
        note['velocity'] = max(VELOCITY_RANGE[0], min(VELOCITY_RANGE[1], note['velocity']))
    elif modification == 'duration':
        # Ajusta la duración asegurando que sea al menos un pulso
        change = random.choice([-BEAT_DURATION, BEAT_DURATION])
        new_end = note['end'] + change
        if new_end > note['start'] + BEAT_DURATION:
            note['end'] = quantize_time(new_end)
        else:
            note['end'] = note['start'] + BEAT_DURATION

    # Asegura que el final sea posterior al inicio
    if note['end'] <= note['start']:
        note['end'] = note['start'] + BEAT_DURATION

    # Cuantiza los tiempos para mantener la alineación
    note['start'] = quantize_time(note['start'])
    note['end'] = quantize_time(note['end'])

    return note_sequence


def mutate(note_sequence: NoteSequence) -> NoteSequence:
    """
    Función de mutación mejorada con operaciones informadas musicalmente.
    Se enfoca en modificar notas existentes y realizar mutaciones estructurales.

    Parámetros:
        note_sequence (NoteSequence): La secuencia actual de notas.

    Retorna:
        NoteSequence: La secuencia de notas mutada.
    """
    mutation_type = random.choices(
        ['pitch_shift', 'time_stretch', 'modify_note', 'add_chord', 'mutate_phrases', 'mutate_motifs'],
        weights=[0.3, 0.2, 0.3, 0.1, 0.05, 0.05],
        k=1
    )[0]
    
    if mutation_type == 'add_chord':
        return add_chord(note_sequence)
    elif mutation_type == 'pitch_shift':
        root = random.choice([60, 62, 64, 65, 67, 69, 71])  # C, D, E, F, G, A, B en MIDI
        scale_type = random.choice(['major', 'minor'])
        semitones = random.choice([-12, -7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 8, 9, 12])
        return pitch_shift_scale_conscious(note_sequence, semitones, root, scale_type)
    elif mutation_type == 'time_stretch':
        factor = random.uniform(0.9, 1.1)
        return time_stretch(note_sequence, factor)
    elif mutation_type == 'modify_note':
        return modify_random_note_improved(note_sequence)
    elif mutation_type == 'mutate_phrases':
        return mutate_phrases(note_sequence)
    elif mutation_type == 'mutate_motifs':
        return mutate_motifs(note_sequence)
    else:
        return note_sequence

# --------------------------- Funciones de Cruza --------------------------- #

def adjust_timing(offspring: NoteSequence) -> NoteSequence:
    """
    Ajusta el tiempo de las notas para mantener un orden cronológico.

    Parámetros:
        offspring (NoteSequence): Secuencia de notas generada.

    Retorna:
        NoteSequence: Secuencia ajustada con tiempos corregidos.
    """
    if not offspring:
        return offspring

    # Ordena las notas por tiempo de inicio
    sorted_offspring = sorted(offspring, key=lambda x: x['start'])
    previous_end = 0.0

    for note in sorted_offspring:
        # Corrige solapamientos para mantener el orden cronológico
        if note['start'] < previous_end:
            note['start'] = previous_end
            note['end'] = note['start'] + (note['end'] - note['start'])
        previous_end = note['end']

    return sorted_offspring

def segment_based_crossover(parent1: NoteSequence, parent2: NoteSequence, segment_length: int = 8) -> Tuple[NoteSequence, NoteSequence]:
    """
    Realiza un cruce basado en segmentos entre dos secuencias de notas, asegurando coherencia musical.

    Parámetros:
        parent1 (NoteSequence): Primera secuencia de notas de los padres.
        parent2 (NoteSequence): Segunda secuencia de notas de los padres.
        segment_length (int): Número de notas consecutivas a intercambiar.

    Retorna:
        Tuple[NoteSequence, NoteSequence]: Dos secuencias resultantes del cruce.
    """
    if not parent1 or not parent2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    # Ordena las secuencias por tiempo de inicio
    sorted_parent1 = sorted(parent1, key=lambda x: x['start'])
    sorted_parent2 = sorted(parent2, key=lambda x: x['start'])

    def get_segments(parent: NoteSequence, seg_length: int) -> List[NoteSequence]:
        # Divide la secuencia en segmentos de longitud fija
        return [parent[i:i + seg_length] for i in range(0, len(parent) - seg_length + 1, seg_length)]

    segments1 = get_segments(sorted_parent1, segment_length)
    segments2 = get_segments(sorted_parent2, segment_length)

    if not segments1 or not segments2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    # Selecciona segmentos aleatorios para intercambiar
    seg1 = random.choice(segments1)
    seg2 = random.choice(segments2)

    def find_segment_start(parent: NoteSequence, segment: NoteSequence) -> int:
        # Encuentra el índice donde comienza un segmento
        for i in range(len(parent) - len(segment) + 1):
            if parent[i:i + len(segment)] == segment:
                return i
        return 0

    start1 = find_segment_start(sorted_parent1, seg1)
    start2 = find_segment_start(sorted_parent2, seg2)

    # Crea los hijos intercambiando los segmentos
    offspring1 = sorted_parent1[:start1] + seg2 + sorted_parent1[start1 + segment_length:]
    offspring2 = sorted_parent2[:start2] + seg1 + sorted_parent2[start2 + segment_length:]

    offspring1 = adjust_timing(offspring1)
    offspring2 = adjust_timing(offspring2)

    # Valida las secuencias resultantes
    offspring1 = validate_note_sequence(offspring1)
    offspring2 = validate_note_sequence(offspring2)

    return offspring1, offspring2

def single_point_crossover(parent1: NoteSequence, parent2: NoteSequence, phrase_length: int = 8, num_crossover_points: int = 1) -> Tuple[NoteSequence, NoteSequence]:
    """
    Realiza un cruce en un único punto en los límites de las frases entre dos secuencias de notas.

    Parámetros:
        parent1 (NoteSequence): Primera secuencia de notas de los padres.
        parent2 (NoteSequence): Segunda secuencia de notas de los padres.
        phrase_length (int): Número de notas por frase.
        num_crossover_points (int): Número de puntos de cruce.

    Retorna:
        Tuple[NoteSequence, NoteSequence]: Dos secuencias resultantes del cruce.
    """
    if not parent1 or not parent2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    # Ordena las secuencias por tiempo de inicio
    sorted_parent1 = sorted(parent1, key=lambda x: x['start'])
    sorted_parent2 = sorted(parent2, key=lambda x: x['start'])

    def get_phrases(parent: NoteSequence, phrase_len: int) -> List[NoteSequence]:
        # Divide las notas en frases de longitud fija
        return [parent[i:i + phrase_len] for i in range(0, len(parent), phrase_len)]

    phrases1 = get_phrases(sorted_parent1, phrase_length)
    phrases2 = get_phrases(sorted_parent2, phrase_length)

    if not phrases1 or not phrases2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    crossover_indices1 = sorted(random.sample(range(1, len(phrases1)), min(num_crossover_points, len(phrases1) - 1)))
    crossover_indices2 = sorted(random.sample(range(1, len(phrases2)), min(num_crossover_points, len(phrases2) - 1)))

    def create_offspring(phrases_a: List[NoteSequence], phrases_b: List[NoteSequence], crossover_points: List[int]) -> NoteSequence:
        # Alterna frases entre los padres para generar descendencia
        offspring = []
        current_source = 'a'
        last_index = 0
        for point in crossover_points + [len(phrases_a)]:
            if current_source == 'a':
                offspring.extend(phrases_a[last_index:point])
                current_source = 'b'
            else:
                offspring.extend(phrases_b[last_index:point])
                current_source = 'a'
            last_index = point
        return offspring

    offspring1_phrases = create_offspring(phrases1, phrases2, crossover_indices1)
    offspring2_phrases = create_offspring(phrases2, phrases1, crossover_indices2)

    offspring1 = [note for phrase in offspring1_phrases for note in phrase]
    offspring2 = [note for phrase in offspring2_phrases for note in phrase]

    offspring1 = adjust_timing(offspring1)
    offspring2 = adjust_timing(offspring2)

    # Valida las secuencias resultantes
    offspring1 = validate_note_sequence(offspring1)
    offspring2 = validate_note_sequence(offspring2)

    return offspring1, offspring2


def uniform_crossover(parent1: NoteSequence, parent2: NoteSequence, swap_probability: float = 0.5) -> Tuple[NoteSequence, NoteSequence]:
    """
    Realiza un cruce uniforme entre dos secuencias de notas, intercambiando notas con una probabilidad definida.

    Parámetros:
        parent1 (NoteSequence): Primera secuencia de notas de los padres.
        parent2 (NoteSequence): Segunda secuencia de notas de los padres.
        swap_probability (float): Probabilidad de intercambiar cada nota (0.0 a 1.0).

    Retorna:
        Tuple[NoteSequence, NoteSequence]: Dos secuencias resultantes del cruce.
    """
    if not parent1 or not parent2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    sorted_parent1 = sorted(parent1, key=lambda x: x['start'])
    sorted_parent2 = sorted(parent2, key=lambda x: x['start'])

    min_length = min(len(sorted_parent1), len(sorted_parent2))

    offspring1 = []
    offspring2 = []

    for i in range(min_length):
        if random.random() < swap_probability:
            offspring1.append(copy.deepcopy(sorted_parent2[i]))
            offspring2.append(copy.deepcopy(sorted_parent1[i]))
        else:
            offspring1.append(copy.deepcopy(sorted_parent1[i]))
            offspring2.append(copy.deepcopy(sorted_parent2[i]))

    # Agrega las notas restantes del padre más largo
    if len(sorted_parent1) > min_length:
        offspring1.extend(copy.deepcopy(sorted_parent1[min_length:]))
    if len(sorted_parent2) > min_length:
        offspring2.extend(copy.deepcopy(sorted_parent2[min_length:]))

    offspring1 = adjust_timing(offspring1)
    offspring2 = adjust_timing(offspring2)

    # Valida las secuencias resultantes
    offspring1 = validate_note_sequence(offspring1)
    offspring2 = validate_note_sequence(offspring2)

    return offspring1, offspring2

def harmonize_offspring(note_sequence: NoteSequence, key: int = 0, scale: List[int] = None) -> NoteSequence:
    """
    Ajusta los tonos y ritmos de una secuencia para asegurar coherencia armónica y rítmica.

    Parámetros:
        note_sequence (NoteSequence): La secuencia de notas generada.
        key (int): Nota raíz de la escala (número MIDI, por ejemplo, 60 para Do central).
        scale (List[int]): Los grados de la escala (intervalos en semitonos desde la raíz). Por defecto, escala mayor en Do.

    Retorna:
        NoteSequence: La secuencia de notas armonizada.
    """
    if scale is None:
        scale = [0, 2, 4, 5, 7, 9, 11]  # Intervalos de la escala mayor en Do

    # Ordena las notas por tiempo de inicio
    sorted_notes = sorted(note_sequence, key=lambda x: x['start'])
    
    previous_pitch = None
    for i, note in enumerate(sorted_notes):
        # **Adherencia a la escala**
        # Ajusta el tono para que encaje en la escala
        relative_pitch = note['pitch'] - key
        scale_degree = relative_pitch % 12
        if scale_degree not in scale:
            # Encuentra el grado de la escala más cercano
            nearest = min(scale, key=lambda sd: abs(sd - scale_degree))
            adjusted_pitch = key + (relative_pitch - scale_degree) + nearest
            note['pitch'] = max(0, min(127, adjusted_pitch))  # Asegura que el tono esté en rango MIDI
        
        # **Preservación del contorno melódico**
        if previous_pitch is not None:
            pitch_diff = note['pitch'] - previous_pitch
            if abs(pitch_diff) > MAX_PITCH_JUMP:
                # Reduce el salto tonal al límite permitido
                pitch_diff = MAX_PITCH_JUMP if pitch_diff > 0 else -MAX_PITCH_JUMP
                note['pitch'] = previous_pitch + pitch_diff
                note['pitch'] = max(0, min(127, note['pitch']))  # Asegura que el tono esté en rango MIDI
        
        previous_pitch = note['pitch']
    
    # **Consistencia rítmica**
    # Alinea los inicios de las notas a una cuadrícula basada en un compás común (por ejemplo, 4/4)
    BEAT_DURATION = 0.5  # Duración del pulso en segundos
    GRID_RESOLUTION = 0.05  # Desviación permitida en segundos

    for note in sorted_notes:
        aligned_start = round(note['start'] / BEAT_DURATION) * BEAT_DURATION
        if abs(note['start'] - aligned_start) > GRID_RESOLUTION:
            note['start'] = aligned_start
            note['end'] = note['start'] + max(0.1, note['end'] - note['start'])  # Asegura que la duración sea positiva
    
    # **Control del rango dinámico**
    # Normaliza las velocidades dentro de un rango preferido
    PREFERRED_VELOCITY_MIN = 60
    PREFERRED_VELOCITY_MAX = 100

    for note in sorted_notes:
        if note['velocity'] < PREFERRED_VELOCITY_MIN:
            note['velocity'] = PREFERRED_VELOCITY_MIN
        elif note['velocity'] > PREFERRED_VELOCITY_MAX:
            note['velocity'] = PREFERRED_VELOCITY_MAX
    
    # **Evita notas superpuestas del mismo tono**
    active_pitches = {}
    for note in sorted_notes:
        pitch = note['pitch']
        if pitch in active_pitches:
            if note['start'] < active_pitches[pitch]:
                # Si hay solapamiento, ajusta el inicio de la nota
                note['start'] = active_pitches[pitch] + 0.1  # Desplaza 0.1 segundos
                note['end'] = note['start'] + max(0.1, note['end'] - note['start'])
        active_pitches[pitch] = note['end']
    
    # Orden final y retorno
    harmonized_sequence = sorted(sorted_notes, key=lambda x: x['start'])
    
    return harmonized_sequence


# --------------------------- Enhanced Fitness Function --------------------------- #

def fitness(note_sequence: list) -> float:
    """
    Función de aptitud mejorada con puntuaciones normalizadas para evitar valores infinitos.
    Incluye penalizaciones por notas fuera de rango y recompensas por intervalos armoniosos.
    También evalúa la consistencia rítmica y el equilibrio dinámico.

    Parámetros:
        note_sequence (list de dict): La secuencia de notas a evaluar.

    Retorna:
        float: Puntuación de aptitud.
    """
    if not note_sequence:
        return 0.0

    # Pesos iniciales para cada componente de la aptitud (la suma debe ser menor a 1.0)
    WEIGHTS = {
        'unique_pitches': 0.18,
        'total_notes': 0.09,
        'smoothness': 0.18,
        'phrase_structure': 0.09,
        'harmony': 0.18,
        'rhythmic_consistency': 0.09,
        'dynamic_balance': 0.19
    }

    try:
        # Ordena las notas por tiempo de inicio
        sorted_notes = sorted(note_sequence, key=lambda x: x['start'])
        total_notes = len(sorted_notes)

        # 1. Puntuación de tonos únicos (0-1)
        unique_pitches = len(set(note['pitch'] for note in sorted_notes))
        unique_pitches_score = min(1.0, unique_pitches / 50.0)  # Normalización ajustable

        # 2. Puntuación total de notas (0-1)
        total_notes_score = min(1.0, total_notes / 500.0)  # Normalización ajustable

        # 3. Puntuación de suavidad (0-1)
        smooth_transitions = 0
        total_transitions = total_notes - 1
        if total_transitions > 0:
            for i in range(total_transitions):
                current_note = sorted_notes[i]
                next_note = sorted_notes[i + 1]

                pitch_diff = abs(current_note['pitch'] - next_note['pitch'])
                time_diff = max(0.0001, next_note['start'] - current_note['end'])

                # Normaliza la diferencia de tonos al intervalo de una octava
                octave_diff = pitch_diff % 12
                if octave_diff > 6:
                    octave_diff = 12 - octave_diff  # Intervalo más pequeño dentro de una octava

                # Define los umbrales
                MAX_PITCH_DIFF = 6  # Media octava
                MAX_TIME_DIFF = 0.5  # Segundos

                if octave_diff <= MAX_PITCH_DIFF and time_diff < MAX_TIME_DIFF:
                    smooth_transitions += 1

            smoothness_score = smooth_transitions / total_transitions
        else:
            smoothness_score = 0.0

        # 4. Puntuación de armonía (0-1)
        harmonious_intervals = 0
        acceptable_intervals = {0, 3, 4, 5, 7, 8, 9, 12}  # Unísonos, terceras, quintas, etc.
        total_intervals = total_transitions
        if total_intervals > 0:
            for i in range(total_transitions):
                current_pitch = sorted_notes[i]['pitch']
                next_pitch = sorted_notes[i + 1]['pitch']
                interval = abs(current_pitch - next_pitch) % 12

                if interval in acceptable_intervals:
                    harmonious_intervals += 1

            harmony_score = harmonious_intervals / total_intervals
        else:
            harmony_score = 0.0

        # 5. Puntuación de estructura de frases (0-1)
        phrase_score = 0.0
        total_possible_phrases = max(1, total_notes // IDEAL_PHRASE_LENGTH)
        phrase_boundaries = 0

        for i in range(1, total_notes):
            time_gap = sorted_notes[i]['start'] - sorted_notes[i - 1]['end']
            # Define un límite razonable para las frases
            if 0.1 <= time_gap <= 0.5:
                phrase_boundaries += 1

        phrase_score = min(1.0, phrase_boundaries / total_possible_phrases)

        # 6. Puntuación de consistencia rítmica (0-1)
        aligned_notes = sum(
            1 for note in sorted_notes
            if abs((note['start'] / BEAT_DURATION) - round(note['start'] / BEAT_DURATION)) < 0.05
        )
        rhythmic_consistency_score = aligned_notes / total_notes if total_notes > 0 else 0.0

        # 7. Puntuación de equilibrio dinámico (0-1)
        velocities = [note['velocity'] for note in sorted_notes]
        mean_velocity = np.mean(velocities) if velocities else 0.0
        dynamic_balance_score = 1 - abs(mean_velocity - PREFERRED_VELOCITY) / PREFERRED_VELOCITY
        dynamic_balance_score = max(0.0, min(1.0, dynamic_balance_score))

        # 8. Penalización por notas fuera de rango
        out_of_range = sum(
            1 for note in sorted_notes if note['pitch'] < MIN_PITCH or note['pitch'] > MAX_PITCH
        )
        if total_notes > 0:
            out_of_range_penalty = 1 - (out_of_range / total_notes)
        else:
            out_of_range_penalty = 1.0

        # 9. Puntuación de legato (0-1)
        legato_score = 0.0
        if total_notes > 1:
            overlapping_notes = 0
            total_pairs = total_notes - 1
            for i in range(total_pairs):
                current_note = sorted_notes[i]
                next_note = sorted_notes[i + 1]
                if current_note['end'] > next_note['start']:
                    overlapping_notes += 1
            legato_score = overlapping_notes / total_pairs if total_pairs > 0 else 0.0

        # 10. Puntuación de consistencia tonal (0-1)
        key, scale_type, key_consistency_score = determine_key(sorted_notes)

        # Actualiza los pesos para incluir los nuevos componentes
        additional_weights = {
            'legato': 0.10,
            'key_consistency': 0.10
        }
        WEIGHTS.update(additional_weights)

        # Normaliza los pesos para que sumen 1.0
        total_weight = sum(WEIGHTS.values())
        WEIGHTS = {k: v / total_weight for k, v in WEIGHTS.items()}

        # Calcula la puntuación final
        final_score = (
            WEIGHTS['unique_pitches'] * unique_pitches_score +
            WEIGHTS['total_notes'] * total_notes_score +
            WEIGHTS['smoothness'] * smoothness_score +
            WEIGHTS['phrase_structure'] * phrase_score +
            WEIGHTS['harmony'] * harmony_score +
            WEIGHTS['rhythmic_consistency'] * rhythmic_consistency_score +
            WEIGHTS['dynamic_balance'] * dynamic_balance_score +
            WEIGHTS['legato'] * legato_score +
            WEIGHTS['key_consistency'] * key_consistency_score
        ) * MAX_SCORE * out_of_range_penalty

        # Asegura que la puntuación final esté en el rango [0, MAX_SCORE]
        final_score = max(0.0, min(MAX_SCORE, final_score))

        return final_score

    except Exception as e:
        print(f"Error en el cálculo de aptitud: {e}")
        return 0.0


# --------------------------- Algoritmo Evolutivo --------------------------- #

def initialize_population(midi_dataset: List[pretty_midi.PrettyMIDI], population_size: int, max_initial_files: int = 500) -> List[NoteSequence]:
    """
    Inicializa la población seleccionando secuencias de notas aleatorias del conjunto de datos MIDI.
    Solo se consideran notas de instrumentos de piano.

    Parámetros:
        midi_dataset (List[pretty_midi.PrettyMIDI]): Conjunto de datos MIDI cargado.
        population_size (int): Tamaño de la población deseada.
        max_initial_files (int): Número máximo de archivos MIDI a usar.

    Retorna:
        List[NoteSequence]: Lista de secuencias de notas inicializadas.
    """
    population = []
    limited_dataset = midi_dataset[:max_initial_files]

    for _ in range(population_size):
        midi = random.choice(limited_dataset)
        combined_notes = []

        # Combina notas solo de instrumentos de piano
        for instrument in midi.instruments:
            if instrument.program == 0 or 'piano' in instrument.name.lower():
                if instrument.notes:
                    notes = midi_to_note_sequence(instrument)
                    combined_notes.extend(notes)

        # Ordena por tiempo de inicio
        if combined_notes:
            combined_notes.sort(key=lambda x: x['start'])
            # Limita la longitud si es necesario
            if len(combined_notes) > 1000:
                combined_notes = combined_notes[:1000]
            population.append(combined_notes)

    return population

def select_best(population: List[NoteSequence], fitnesses: List[float], selection_size: int) -> List[NoteSequence]:
    """
    Selecciona los individuos con mejor rendimiento según las puntuaciones de aptitud,
    incluyendo verificaciones de seguridad.

    Parámetros:
        population (List[NoteSequence]): Población actual de secuencias.
        fitnesses (List[float]): Puntuaciones de aptitud correspondientes.
        selection_size (int): Número de individuos a seleccionar.

    Retorna:
        List[NoteSequence]: Lista de secuencias seleccionadas.
    """
    if not population or not fitnesses:
        return []

    # Elimina valores infinitos o NaN
    valid_pairs = [(p, f) for p, f in zip(population, fitnesses) if f != float('inf') and not np.isnan(f)]

    if not valid_pairs:
        return random.sample(population, min(selection_size, len(population)))

    # Ordena según las puntuaciones de aptitud
    paired_sorted = sorted(valid_pairs, key=lambda x: x[1], reverse=True)

    # Selecciona los mejores individuos
    selected = [individual for individual, score in paired_sorted[:selection_size]]

    # Completa con individuos aleatorios si faltan
    while len(selected) < selection_size and len(population) > len(selected):
        remaining = [p for p in population if p not in selected]
        selected.append(random.choice(remaining))

    return selected


def create_next_generation(selected: List[NoteSequence],
                           population_size: int,
                           mutation_rate: float,
                           crossover_rate: float = 0.8,
                           crossover_types: Dict[str, float] = {'single_point': 0.33, 'segment_based': 0.33, 'uniform': 0.34}) -> List[NoteSequence]:
    """
    Crea la próxima generación usando cruce y mutación.
    Limita la densidad de notas para evitar secuencias sobrecargadas.

    Parámetros:
        selected (List[NoteSequence]): Individuos seleccionados con mejor rendimiento.
        population_size (int): Tamaño deseado de la población.
        mutation_rate (float): Probabilidad de aplicar mutación.
        crossover_rate (float): Probabilidad de aplicar cruce.
        crossover_types (Dict[str, float]): Tipos de cruce con sus probabilidades respectivas.

    Retorna:
        List[NoteSequence]: La próxima generación de secuencias de notas.
    """
    next_generation = []
    max_attempts = population_size * 2
    attempts = 0

    while len(next_generation) < population_size and attempts < max_attempts:
        parent1, parent2 = random.sample(selected, 2)

        # Aplica cruce
        if random.random() < crossover_rate:
            if crossover_types == 'single_point':
                offspring1, offspring2 = single_point_crossover(parent1, parent2)
            elif crossover_types == 'segment_based':
                offspring1, offspring2 = segment_based_crossover(parent1, parent2)
            elif crossover_types == 'uniform':
                offspring1, offspring2 = uniform_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        else:
            offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Armoniza la descendencia
        offspring1 = harmonize_offspring(offspring1)
        offspring2 = harmonize_offspring(offspring2)

        # Aplica mutación
        if random.random() < mutation_rate:
            offspring1 = mutate(offspring1)
        if random.random() < mutation_rate:
            offspring2 = mutate(offspring2)

        # Valida y cuantiza
        offspring1 = validate_note_sequence(offspring1)
        offspring2 = validate_note_sequence(offspring2)

        # Aplica mejoras musicales
        for offspring in [offspring1, offspring2]:
            offspring = smooth_velocities(offspring)
            offspring = apply_phrase_dynamics(offspring)
            offspring = apply_accentuation(offspring)
            offspring = apply_voice_leading(offspring)
            offspring = apply_swing(offspring)
            offspring = apply_dynamic_balance(offspring)

            if offspring:
                next_generation.append(offspring)

        attempts += 1

    # Completa con copias mutadas si faltan individuos
    while len(next_generation) < population_size and selected:
        offspring = copy.deepcopy(random.choice(selected))
        if random.random() < mutation_rate:
            offspring = mutate(offspring)
        offspring = validate_note_sequence(offspring)
        offspring = smooth_velocities(offspring)
        offspring = apply_phrase_dynamics(offspring)
        offspring = apply_accentuation(offspring)
        offspring = apply_voice_leading(offspring)
        offspring = apply_swing(offspring)
        offspring = apply_dynamic_balance(offspring)
        next_generation.append(offspring)

    return next_generation

def evolutionary_music_generation(midi_dataset: List[pretty_midi.PrettyMIDI],
                                  generations: int,
                                  population_size: int,
                                  selection_size: int,
                                  mutation_rate: float,
                                  crossover_rate: float = 0.8,
                                  crossover_types: Dict[str, float] = {'single_point': 0.33, 'segment_based': 0.33, 'uniform': 0.34}) -> List[NoteSequence]:
    """
    Ejecuta el algoritmo evolutivo con cruce y mutación.

    Parámetros:
        midi_dataset (List[pretty_midi.PrettyMIDI]): Conjunto de datos MIDI cargado.
        generations (int): Número de generaciones a evolucionar.
        population_size (int): Tamaño de la población en cada generación.
        selection_size (int): Número de individuos seleccionados por generación.
        mutation_rate (float): Probabilidad de aplicar mutación.
        crossover_rate (float): Probabilidad de aplicar cruce.
        crossover_types (Dict[str, float]): Tipos de cruce y sus probabilidades.

    Retorna:
        NoteSequence: La mejor secuencia evolucionada.
    """
    try:
        population = initialize_population(midi_dataset, population_size)
        if not population:
            logging.error("Falló la inicialización de la población")
            return []

        best_individual = None
        best_fitness = float('-inf')

        for generation in range(1, generations + 1):
            try:
                # Calcula las puntuaciones de aptitud
                fitnesses = [fitness(ind) if ind else 0.0 for ind in population]

                # Actualiza el mejor individuo
                current_best = max(fitnesses, default=0.0)
                if current_best > best_fitness:
                    best_fitness = current_best
                    best_individual = population[fitnesses.index(current_best)]

                # Selección y generación de nueva población
                selected = select_best(population, fitnesses, selection_size)
                population = create_next_generation(selected, population_size, mutation_rate, crossover_rate, crossover_types)

            except Exception as e:
                logging.error(f"Error en la generación {generation}: {e}")
                population = initialize_population(midi_dataset, population_size)

        return best_individual if best_individual else population[0] if population else []

    except Exception as e:
        logging.fatal(f"Error fatal en la generación de música evolutiva: {e}")
        return []

# --------------------------- Main --------------------------- #

def main():
    # Paso 1: Cargar archivos MIDI
    print("Cargando archivos MIDI...")
    # Para pruebas iniciales, limita la cantidad de archivos para acelerar la carga
    midi_dataset = load_midi_files_parallel(DATASET_PATH, max_files=MAX_INITIAL_FILES, log_file_path=LOG_FILE_PATH)
    if not midi_dataset:
        print("No se cargaron archivos MIDI. Saliendo del programa.")
        return

    # Paso 2: Ejecutar el algoritmo evolutivo
    print("Creando música excepcional.")
    best_music = evolutionary_music_generation(
        midi_dataset=midi_dataset,
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        selection_size=SELECTION_SIZE,
        mutation_rate=MUTATION_RATE
    )

    # Paso 3: Convertir el mejor individuo a MIDI y guardar
    print("Convirtiendo el mejor individuo a MIDI...")
    #best_music = eliminate_long_silences(best_music, max_silence_duration=1.0)  # (opcional) Elimina silencios largos
    #best_music = remove_long_single_notes(best_music, max_note_duration=1.0)  # (opcional) Ajusta notas largas
    output_midi = note_sequence_to_midi(best_music)
    output_midi.write(OUTPUT_MIDI_PATH)
    print(f"Música evolucionada guardada como '{OUTPUT_MIDI_PATH}'.")

if __name__ == "__main__":
    main()
