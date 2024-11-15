import os
import pretty_midi
import random
import copy
import numpy as np
import random
from typing import List, Dict, Any
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

# --------------------------- Configuration --------------------------- #

# Path to your dataset
DATASET_PATH = '/Users/jaimeuria/.cache/kagglehub/datasets/imsparsh/lakh-midi-clean/versions/1'

# Evolutionary Algorithm Parameters
POPULATION_SIZE = 100        # Number of individuals in the population
SELECTION_SIZE = 50         # Number of top individuals to select
GENERATIONS = 30          # Number of generations to evolve
MUTATION_RATE = 0.8         # Probability of mutation for each individual
OUTPUT_MIDI_PATH = 'evolved_music.mid'  # Path to save the evolved MIDI file
LOG_FILE_PATH = 'midi_load_log.txt'     # Path to save the MIDI file load log

# Mutation Parameters
PITCH_SHIFT_RANGE = (-12, 12)   # Semitones for pitch shifting
TIME_STRETCH_FACTOR = (0.8, 1.2)  # Factor range for time stretching
MAX_PITCH = 127
MAX_TIME = 150
VELOCITY_RANGE = (30, 100)

# Maximum number of MIDI files to load initially (for testing)
MAX_INITIAL_FILES = 500
TEMPO = random.uniform(100, 150)

# --------------------------- Data Structures --------------------------- #

# Define the structure for a musical note
Note = Dict[str, Any]
NoteSequence = List[Note]

# --------------------------- Helper Functions --------------------------- #

def load_single_midi(midi_path: str) -> Optional[pretty_midi.PrettyMIDI]:
    """
    Attempt to load a single MIDI file.
    Returns the PrettyMIDI object if successful, else None.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        return midi
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return None
    
def load_midi_with_path(path: str) -> Tuple[str, Optional[pretty_midi.PrettyMIDI]]:
    midi = load_single_midi(path)
    return (path, midi)

def load_midi_files_parallel(dataset_path: str, max_files: int = None, log_file_path: str = None) -> List[pretty_midi.PrettyMIDI]:
    """
    Load MIDI files using parallel processing.
    Returns a list of successfully loaded PrettyMIDI objects.
    Logs and skips files that cause errors.
    Optionally limits the number of MIDI files loaded.
    Writes the file paths of the loaded MIDI files to a log file.
    """
    midi_files = []
    band_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    total_bands = len(band_dirs)
    print(f"Found {total_bands} band directories.")

    # Collect all MIDI file paths
    all_midi_paths = []
    for band_dir in band_dirs:
        band_path = os.path.join(dataset_path, band_dir)
        midi_files_in_band = [f for f in os.listdir(band_path) if f.lower().endswith(('.mid', '.midi'))]
        for file in midi_files_in_band:
            midi_path = os.path.join(band_path, file)
            all_midi_paths.append(midi_path)
            if max_files and len(all_midi_paths) >= max_files:
                break
        if max_files and len(all_midi_paths) >= max_files:
            break

    total_files = len(all_midi_paths)
    print(f"Total MIDI files to load: {total_files}")

    # Shuffle the list to randomize file selection
    random.shuffle(all_midi_paths)
    
    # Use multiprocessing Pool to load files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = tqdm(pool.imap_unordered(load_midi_with_path, all_midi_paths), total=total_files, desc="Loading MIDI Files")
        for path, midi in results:
            if midi is not None:
                midi_files.append(midi)
                if log_file_path:
                    # Extract the last folder and filename
                    parent_dir = os.path.basename(os.path.dirname(path))
                    filename = os.path.basename(path)
                    relative_path = os.path.join(parent_dir, filename)
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"{relative_path}\n")
            else:
                # Errors are already printed in load_single_midi
                pass

    print(f"Successfully loaded {len(midi_files)} MIDI files.")
    return midi_files

def midi_to_note_sequence(instrument: pretty_midi.Instrument) -> NoteSequence:
    """
    Convert a pretty_midi.Instrument object to a note sequence.
    """
    notes = []
    for note in instrument.notes:
        notes.append({
            'pitch': note.pitch,
            'start': note.start,
            'end': note.end,
            'velocity': note.velocity
        })
    # Sort notes by start time
    notes.sort(key=lambda x: x['start'])
    return notes

def note_sequence_to_midi(note_sequence: NoteSequence, tempo: float = TEMPO) -> pretty_midi.PrettyMIDI:
    """
    Convert a note sequence back to a PrettyMIDI object.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    for note in note_sequence:
        midi_note = pretty_midi.Note(
            velocity=int(note['velocity']),
            pitch=int(note['pitch']),
            start=float(note['start']),
            end=float(note['end'])
        )
        instrument.notes.append(midi_note)
    midi.instruments.append(instrument)
    return midi

# --------------------------- Mutation Operators --------------------------- #

def pitch_shift(note_sequence: List[List[Dict]], semitones: int) -> List[List[Dict]]:
    """
    Transpose all notes in the sequence by a given number of semitones.
    """
    shifted = copy.deepcopy(note_sequence)
    for instrument_notes in shifted:
        for note in instrument_notes:
            note['pitch'] += semitones
            # Ensure pitch stays within MIDI range
            note['pitch'] = max(0, min(MAX_PITCH, note['pitch']))
    return shifted

def time_stretch(note_sequence: List[List[Dict]], factor: float) -> List[List[Dict]]:
    """
    Stretch the timing of all notes by a given factor.
    """
    stretched = copy.deepcopy(note_sequence)
    for instrument_notes in stretched:
        for note in instrument_notes:
            note['start'] *= factor
            note['end'] *= factor
            # Ensure start and end times stay within bounds
            note['start'] = max(0, note['start'])
            note['end'] = max(note['start'] + 0.1, note['end'])  # Ensure duration >= 0.1
    return stretched

def add_random_note(note_sequence: List[List[Dict]]) -> List[List[Dict]]:
    """
    Add a random note to a random instrument in the sequence.
    The added note adheres to the C major key to enhance key adherence.
    """
    shifted = copy.deepcopy(note_sequence)
    instrument_idx = random.randint(0, len(shifted) - 1)
    instrument_notes = shifted[instrument_idx]
    
    # C major pitch classes: C, D, E, F, G, A, B => MIDI: 0, 2, 4, 5, 7, 9, 11 modulo 12
    c_major_pitch_classes = [0, 2, 4, 5, 7, 9, 11]
    base_pitch = random.choice(c_major_pitch_classes) + random.randint(0, 10) * 12  # Spread across octaves
    new_pitch = max(0, min(MAX_PITCH, base_pitch))
    
    new_start = random.uniform(0, MAX_TIME)
    new_end = new_start + random.uniform(0.1, 1.0)  # Ensure duration >= 0.1
    new_velocity = random.randint(VELOCITY_RANGE[0], VELOCITY_RANGE[1])
    
    new_note = {
        'pitch': new_pitch,
        'start': new_start,
        'end': new_end,
        'velocity': new_velocity
    }
    
    instrument_notes.append(new_note)
    # Sort by start time
    instrument_notes.sort(key=lambda x: x['start'])
    return shifted

def delete_random_note(note_sequence: List[List[Dict]]) -> List[List[Dict]]:
    """
    Delete a random note from a random instrument in the sequence.
    Preferentially deletes notes with lower velocities to maintain musicality.
    """
    shifted = copy.deepcopy(note_sequence)
    # Filter instruments that have notes
    instruments_with_notes = [i for i, inst in enumerate(shifted) if inst]
    if not instruments_with_notes:
        return shifted
    instrument_idx = random.choice(instruments_with_notes)
    instrument_notes = shifted[instrument_idx]
    
    # Prefer deleting notes with lower velocity
    low_velocity_notes = [i for i, note in enumerate(instrument_notes) if note['velocity'] < 60]
    if low_velocity_notes:
        idx = random.choice(low_velocity_notes)
    else:
        idx = random.randint(0, len(instrument_notes) - 1)
    del instrument_notes[idx]
    return shifted

def modify_random_note(note_sequence: List[List[Dict]]) -> List[List[Dict]]:
    """
    Modify a random attribute of a random note in a random instrument in the sequence.
    """
    shifted = copy.deepcopy(note_sequence)
    # Filter instruments that have notes
    instruments_with_notes = [i for i, inst in enumerate(shifted) if inst]
    if not instruments_with_notes:
        return shifted
    instrument_idx = random.choice(instruments_with_notes)
    instrument_notes = shifted[instrument_idx]
    idx = random.randint(0, len(instrument_notes) - 1)
    note = instrument_notes[idx]
    
    modification = random.choice(['pitch', 'velocity', 'start', 'end'])
    if modification == 'pitch':
        # Shift pitch within C major scale
        c_major_pitch_classes = [0, 2, 4, 5, 7, 9, 11]
        current_pitch_class = note['pitch'] % 12
        possible_shifts = [p for p in c_major_pitch_classes if p != current_pitch_class]
        if possible_shifts:
            pitch_shift = random.choice(possible_shifts) - current_pitch_class
            note['pitch'] += pitch_shift
            note['pitch'] = max(0, min(MAX_PITCH, note['pitch']))
    elif modification == 'velocity':
        velocity_change = random.choice([-10, -5, 5, 10])
        note['velocity'] += velocity_change
        note['velocity'] = max(VELOCITY_RANGE[0], min(VELOCITY_RANGE[1], note['velocity']))
    elif modification == 'start':
        time_change = random.uniform(-0.2, 0.2)
        note['start'] += time_change
        note['start'] = max(0, note['start'])
        if note['end'] <= note['start']:
            note['end'] = note['start'] + 0.1
    elif modification == 'end':
        time_change = random.uniform(-0.2, 0.2)
        note['end'] += time_change
        note['end'] = max(note['start'] + 0.1, note['end'])
    # Sort by start time after modification
    instrument_notes.sort(key=lambda x: x['start'])
    return shifted

def add_harmonic_note(note_sequence: List[List[Dict]]) -> List[List[Dict]]:
    """
    Add a note that forms a chord with existing notes to enhance harmonic richness.
    """
    shifted = copy.deepcopy(note_sequence)
    # Choose a random instrument to add the harmonic note
    instrument_idx = random.randint(0, len(shifted) - 1)
    instrument_notes = shifted[instrument_idx]
    
    if not instrument_notes:
        return shifted  # No notes to harmonize with
    
    # Choose a random note to harmonize with
    base_note = random.choice(instrument_notes)
    base_pitch = base_note['pitch']
    
    # Define chord intervals (e.g., major triad: +4, +7 semitones)
    chord_intervals = [4, 7]  # Major chord
    for interval in chord_intervals:
        harmonic_pitch = base_pitch + interval
        if harmonic_pitch > MAX_PITCH:
            continue
        new_start = base_note['start']
        new_end = base_note['end']
        new_velocity = base_note['velocity']
        
        new_note = {
            'pitch': harmonic_pitch,
            'start': new_start,
            'end': new_end,
            'velocity': new_velocity
        }
        # Ensure key adherence (C major)
        if harmonic_pitch % 12 not in [0, 2, 4, 5, 7, 9, 11]:
            continue
        instrument_notes.append(new_note)
    
    # Sort by start time
    instrument_notes.sort(key=lambda x: x['start'])
    return shifted

def melodic_smoothing(note_sequence: List[List[Dict]]) -> List[List[Dict]]:
    """
    Smooth melodic transitions by minimizing large intervals between consecutive notes.
    """
    shifted = copy.deepcopy(note_sequence)
    for instrument_notes in shifted:
        # Sort notes by start time
        sorted_notes = sorted(instrument_notes, key=lambda x: x['start'])
        for i in range(1, len(sorted_notes)):
            prev_pitch = sorted_notes[i-1]['pitch']
            current_pitch = sorted_notes[i]['pitch']
            interval = current_pitch - prev_pitch
            if abs(interval) > 4:  # Threshold for large intervals
                # Adjust current pitch to reduce the interval
                adjustment = -2 if interval > 0 else 2
                new_pitch = current_pitch + adjustment
                new_pitch = max(0, min(MAX_PITCH, new_pitch))
                # Ensure new pitch adheres to C major
                if new_pitch % 12 in [0, 2, 4, 5, 7, 9, 11]:
                    sorted_notes[i]['pitch'] = new_pitch
        # Update instrument notes
        instrument_notes[:] = sorted_notes
    return shifted

def rhythmic_variation(note_sequence: List[List[Dict]]) -> List[List[Dict]]:
    """
    Introduce rhythmic variations by altering note durations and start times.
    """
    shifted = copy.deepcopy(note_sequence)
    for instrument_notes in shifted:
        for note in instrument_notes:
            # Randomly decide whether to alter this note
            if random.random() < 0.3:  # 30% chance to alter
                duration = note['end'] - note['start']
                duration_change = random.uniform(-0.1, 0.1)
                new_duration = max(0.1, duration + duration_change)
                note['end'] = note['start'] + new_duration
    return shifted

def adjust_polyphony(note_sequence: List[List[Dict]], target_polyphony: int) -> List[List[Dict]]:
    """
    Adjust the polyphony of the note sequence to approach the target polyphony.
    """
    shifted = copy.deepcopy(note_sequence)
    current_polyphony = calculate_max_polyphony(shifted)
    if current_polyphony < target_polyphony:
        # Add simultaneous notes
        for instrument_notes in shifted:
            if not instrument_notes:
                continue
            # Choose a random note to duplicate
            base_note = random.choice(instrument_notes)
            new_pitch = base_note['pitch'] + random.choice([-2, 2])  # Add a third interval
            if 0 <= new_pitch <= MAX_PITCH and new_pitch % 12 in [0, 2, 4, 5, 7, 9, 11]:
                new_note = {
                    'pitch': new_pitch,
                    'start': base_note['start'],
                    'end': base_note['end'],
                    'velocity': base_note['velocity']
                }
                instrument_notes.append(new_note)
    elif current_polyphony > target_polyphony:
        # Remove overlapping notes
        for instrument_notes in shifted:
            # Sort notes by start time
            sorted_notes = sorted(instrument_notes, key=lambda x: x['start'])
            active_notes = []
            for note in sorted_notes:
                # Remove notes that have ended
                active_notes = [n for n in active_notes if n['end'] > note['start']]
                if len(active_notes) >= target_polyphony:
                    # Remove the current note
                    note['pitch'] = None  # Mark for deletion
                else:
                    active_notes.append(note)
            # Remove marked notes
            instrument_notes[:] = [n for n in sorted_notes if n['pitch'] is not None]
    return shifted

def calculate_max_polyphony(note_sequence: List[List[Dict]]) -> int:
    """
    Calculate the maximum number of overlapping notes (polyphony) in the sequence.
    """
    all_notes = [note for instrument_notes in note_sequence for note in instrument_notes]
    if not all_notes:
        return 0
    events = []
    for note in all_notes:
        events.append((note['start'], 'start'))
        events.append((note['end'], 'end'))
    events.sort()
    
    current_polyphony = 0
    max_polyphony = 0
    for time, event in events:
        if event == 'start':
            current_polyphony += 1
            if current_polyphony > max_polyphony:
                max_polyphony = current_polyphony
        else:
            current_polyphony -= 1
    return max_polyphony

def repeat_motif(note_sequence: List[List[Dict]], motif_length: int = 4) -> List[List[Dict]]:
    """
    Repeat a randomly selected motif within the note sequence to balance repetition and variation.
    """
    shifted = copy.deepcopy(note_sequence)
    # Choose a random instrument
    instrument_idx = random.randint(0, len(shifted) - 1)
    instrument_notes = shifted[instrument_idx]
    if len(instrument_notes) < motif_length:
        return shifted  # Not enough notes to form a motif
    # Choose a random motif
    start_idx = random.randint(0, len(instrument_notes) - motif_length)
    motif = instrument_notes[start_idx:start_idx + motif_length]
    
    # Calculate the time shift to place the motif later in the sequence
    last_end_time = max(note['end'] for note in instrument_notes)
    time_shift = random.uniform(1.0, 5.0)  # Shift motif by 1 to 5 seconds/beats
    
    # Duplicate the motif with time shift
    new_motif = []
    for note in motif:
        new_note = copy.deepcopy(note)
        new_note['start'] += time_shift
        new_note['end'] += time_shift
        # Ensure times are within bounds
        new_note['start'] = min(new_note['start'], MAX_TIME - 0.1)
        new_note['end'] = min(new_note['end'], MAX_TIME)
        new_motif.append(new_note)
    
    instrument_notes.extend(new_motif)
    # Sort by start time
    instrument_notes.sort(key=lambda x: x['start'])
    return shifted

def crossover(parent1: List[List[Dict]], parent2: List[List[Dict]]) -> List[List[Dict]]:
    """
    Combine two parent note sequences to produce an offspring note sequence.
    Uses single-point crossover for each instrument.
    """
    offspring = []
    for inst1, inst2 in zip(parent1, parent2):
        if not inst1 and not inst2:
            offspring.append([])
            continue
        # Choose crossover point based on the number of notes
        len1 = len(inst1)
        len2 = len(inst2)
        crossover_point1 = random.randint(0, len1) if len1 > 0 else 0
        crossover_point2 = random.randint(0, len2) if len2 > 0 else 0
        
        # Create offspring instrument notes by combining parts from both parents
        new_instrument = inst1[:crossover_point1] + inst2[crossover_point2:]
        
        # Sort by start time to maintain temporal structure
        new_instrument.sort(key=lambda x: x['start'])
        offspring.append(new_instrument)
    return offspring



def mutate(note_sequence: List[List[Dict]]) -> List[List[Dict]]:
    """
    Apply a random mutation to the note sequence, biased towards improving fitness.
    """
    mutation_type = random.choices(
        ['pitch_shift', 'time_stretch', 'add_note', 'delete_note', 'modify_note'],
        weights=[2, 2, 3, 1, 2],  # Adjust weights to favor certain mutations
        k=1
    )[0]
    
    if mutation_type == 'pitch_shift':
        semitones = random.randint(*PITCH_SHIFT_RANGE)
        return pitch_shift(note_sequence, semitones)
    elif mutation_type == 'time_stretch':
        factor = random.uniform(*TIME_STRETCH_FACTOR)
        return time_stretch(note_sequence, factor)
    elif mutation_type == 'add_note':
        return add_random_note(note_sequence)
    elif mutation_type == 'delete_note':
        return delete_random_note(note_sequence)
    elif mutation_type == 'modify_note':
        return modify_random_note(note_sequence)
    else:
        return note_sequence


# --------------------------- Fitness Function --------------------------- #

def fitness(note_sequence: List[List[Dict]]) -> float:
    """
    Calculate the fitness of a note sequence.
    Higher fitness indicates a "better" musical piece.
    """
    if not any(note_sequence):
        return 0.0

    # Flatten all notes across all instruments
    all_notes = [note for instrument_notes in note_sequence for note in instrument_notes]
    if not all_notes:
        return 0.0

    # Basic Metrics
    unique_pitches = len(set(note['pitch'] for note in all_notes))
    total_notes = len(all_notes)
    total_duration = sum(note['end'] - note['start'] for note in all_notes)

    # Pitch Range
    pitches = [note['pitch'] for note in all_notes]
    pitch_range = max(pitches) - min(pitches) if pitches else 0

    # Average Interval
    sorted_notes = sorted(all_notes, key=lambda x: x['start'])
    intervals = [
        abs(sorted_notes[i+1]['pitch'] - sorted_notes[i]['pitch'])
        for i in range(len(sorted_notes) - 1)
    ]
    avg_interval = np.mean(intervals) if intervals else 0

    # Rhythmic Diversity
    durations = [round(note['end'] - note['start'], 2) for note in all_notes]
    unique_durations = len(set(durations))
    rhythmic_diversity = unique_durations / total_notes if total_notes else 0

    # Harmonic Richness
    # Simplistic approach: count unique pitch classes
    pitch_classes = set(p % 12 for p in pitches)
    harmonic_richness = len(pitch_classes) / 12  # Normalize between 0 and 1

    # Polyphony
    # Create a list of all note start and end times
    events = []
    for note in all_notes:
        events.append((note['start'], 'start'))
        events.append((note['end'], 'end'))
    events.sort()
    
    current_polyphony = 0
    max_polyphony = 0
    for time, event in events:
        if event == 'start':
            current_polyphony += 1
            if current_polyphony > max_polyphony:
                max_polyphony = current_polyphony
        else:
            current_polyphony -= 1
    polyphony = max_polyphony

    # Repetition vs. Variation
    motifs = {}
    motif_length = 4  # Number of consecutive pitches to consider as a motif
    for i in range(len(pitches) - motif_length + 1):
        motif = tuple(pitches[i:i+motif_length])
        motifs[motif] = motifs.get(motif, 0) + 1
    repeated_motifs = sum(1 for count in motifs.values() if count > 1)
    repetition_ratio = repeated_motifs / len(motifs) if motifs else 0

    # Tempo Consistency
    # Calculate inter-onset intervals (time between consecutive note starts)
    sorted_notes = sorted(all_notes, key=lambda x: x['start'])
    onset_times = [note['start'] for note in sorted_notes]
    inter_onset_intervals = [
        onset_times[i+1] - onset_times[i]
        for i in range(len(onset_times) - 1)
    ]
    tempo_variance = np.var(inter_onset_intervals) if inter_onset_intervals else 0
    tempo_consistency = 1 / (1 + tempo_variance)  # Higher is better

    # Instrument Balance
    instrument_note_counts = [len(instrument_notes) for instrument_notes in note_sequence]
    if instrument_note_counts:
        instrument_balance = 1 - (np.std(instrument_note_counts) / max(instrument_note_counts))
    else:
        instrument_balance = 1

    # Key Adherence (Assuming C major for simplicity)
    # Notes in C major: C, D, E, F, G, A, B (MIDI pitches: 60, 62, 64, 65, 67, 69, 71, etc.)
    c_major_pitch_classes = {0, 2, 4, 5, 7, 9, 11}
    adherence = sum(1 for p in pitch_classes if p in c_major_pitch_classes) / len(pitch_classes) if pitch_classes else 1

    # Melodic Contour
    contour_changes = 0
    for i in range(len(pitches) - 1):
        current_direction = np.sign(pitches[i+1] - pitches[i])
        previous_direction = np.sign(pitches[i] - pitches[i-1]) if i > 0 else 0
        if current_direction != previous_direction and current_direction != 0:
            contour_changes += 1
    melodic_contour = contour_changes / len(pitches) if pitches else 0

    # Normalizing Metrics
    # Define weights for each metric
    weights = {
        'unique_pitches': 1.0,
        'total_notes': 0.1,
        'pitch_range': 0.05,
        'avg_interval': 0.05,
        'rhythmic_diversity': 1.0,
        'harmonic_richness': 1.0,
        'polyphony': 0.5,
        'repetition_ratio': 0.5,
        'tempo_consistency': 1.0,
        'instrument_balance': 1.0,
        'adherence': 1.0,
        'melodic_contour': 0.5
    }

    # Scale metrics to a comparable range (0-1)
    # You may need to adjust these scaling factors based on your dataset
    max_pitch_range = 24  # Two octaves
    max_unique_pitches = 128  # MIDI range
    max_total_notes = 1000  # Arbitrary large number
    max_polyphony = 10  # Arbitrary max polyphony

    scaled_metrics = {
        'unique_pitches': min(unique_pitches / max_unique_pitches, 1.0),
        'total_notes': min(total_notes / max_total_notes, 1.0),
        'pitch_range': min(pitch_range / max_pitch_range, 1.0),
        'avg_interval': min(avg_interval / 12, 1.0),  # Assuming max interval of an octave
        'rhythmic_diversity': rhythmic_diversity,  # Already between 0 and 1
        'harmonic_richness': harmonic_richness,    # Already between 0 and 1
        'polyphony': min(polyphony / max_polyphony, 1.0),
        'repetition_ratio': repetition_ratio,      # Between 0 and 1
        'tempo_consistency': tempo_consistency,    # Between 0 and 1
        'instrument_balance': instrument_balance,  # Between 0 and 1
        'adherence': adherence,                    # Between 0 and 1
        'melodic_contour': melodic_contour         # Between 0 and 1
    }

    # Compute weighted sum
    fitness_score = sum(weights[metric] * scaled_metrics[metric] for metric in weights)

    return fitness_score

# --------------------------- Evolutionary Algorithm --------------------------- #

def initialize_population(
    midi_dataset: List[pretty_midi.PrettyMIDI],
    population_size: int,
    max_initial_files: int = 500
) -> List[List[List[Dict]]]:
    """
    Initialize the population with random note sequences from the dataset.
    Optionally limit the number of MIDI files to load.
    """
    population = []
    limited_dataset = midi_dataset[:max_initial_files]  # Limit the dataset
    for _ in range(population_size):
        midi = random.choice(limited_dataset)
        note_sequence = [midi_to_note_sequence(instrument) for instrument in midi.instruments]
        # Optionally limit the number of notes per instrument to manage complexity
        for i in range(len(note_sequence)):
            if len(note_sequence[i]) > 1000:
                note_sequence[i] = note_sequence[i][:1000]
        population.append(note_sequence)
    return population


def select_best(population: List[List[NoteSequence]], fitnesses: List[float], selection_size: int) -> List[List[NoteSequence]]:
    """
    Select the top-performing individuals based on fitness scores.
    """
    # Pair each individual with its fitness
    paired = list(zip(population, fitnesses))
    # Sort based on fitness in descending order
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
    # Select the top individuals
    selected = [individual for individual, score in paired_sorted[:selection_size]]
    return selected

def create_next_generation(
    selected: List[List[List[Dict]]],
    population_size: int,
    mutation_rate: float,
    elitism_count: int = 2
) -> List[List[List[Dict]]]:
    """
    Create the next generation by combining selected individuals through crossover and applying mutations.
    Implements elitism by carrying forward top individuals unchanged.
    """
    next_generation = []
    
    # Elitism: Carry forward the top individuals unchanged
    elites = selected[:elitism_count]
    next_generation.extend(copy.deepcopy(elites))
    
    while len(next_generation) < population_size:
        # Selection: Choose two parents
        if len(selected) < 2:
            parent1 = parent2 = random.choice(selected)
        else:
            parent1, parent2 = random.sample(selected, 2)
        
        # Crossover: Combine parents to create offspring
        offspring = crossover(parent1, parent2)
        
        # Mutation: Apply mutation based on the mutation rate
        if random.random() < mutation_rate:
            offspring = mutate(offspring)
        
        next_generation.append(offspring)
    
    return next_generation



def note_sequence_to_multi_track_midi(note_sequence: List[NoteSequence], tempo: float = TEMPO) -> pretty_midi.PrettyMIDI:
    """
    Convert a multi-track note sequence back to a PrettyMIDI object with different instruments.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument_programs = [
        0, 1,  # Acoustic Grand Piano, Bright Acoustic Piano
        24, 25, 26,  # Electric Guitar (clean), Electric Guitar (muted), Overdriven Guitar
        32, 33, 34,  # Acoustic Bass, Electric Bass (finger), Electric Bass (pick)
        40, 41, 42,  # Violin, Viola, Cello
        56, 57, 71,  # Flute, Oboe, Trumpet
        73, 80, 81,  # French Horn, Soprano Sax, Alto Sax
        116, 117, 118  # Drum Kit, SFX 1, SFX 2
    ]
    
    for i, instrument_notes in enumerate(note_sequence):
        instrument = pretty_midi.Instrument(program=instrument_programs[i % len(instrument_programs)])
        for note in instrument_notes:
            midi_note = pretty_midi.Note(
                velocity=int(note['velocity']),
                pitch=int(note['pitch']),
                start=float(note['start']),
                end=float(note['end'])
            )
            instrument.notes.append(midi_note)
        midi.instruments.append(instrument)
    
    return midi

def evolutionary_music_generation(
    midi_dataset: List[pretty_midi.PrettyMIDI],
    generations: int,
    population_size: int,
    selection_size: int,
    mutation_rate: float
) -> List[List[Dict]]:
    initial_mutation_rate = mutation_rate
    final_mutation_rate = mutation_rate

    """
    Run the evolutionary algorithm to generate music with dynamic mutation rates.
    """
    population = initialize_population(midi_dataset, population_size)
    
    for generation in range(1, generations + 1):
        # Evaluate fitness
        fitnesses = [fitness(individual) for individual in population]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
        
        # Selection
        selected = select_best(population, fitnesses, selection_size)
        
        # Dynamic Mutation Rate
        mutation_rate = initial_mutation_rate + (final_mutation_rate - initial_mutation_rate) * (generation / generations)
        
        # Create next generation with crossover and mutation
        population = create_next_generation(selected, population_size, mutation_rate)
    
    # After all generations, select the best individual
    final_fitnesses = [fitness(individual) for individual in population]
    best_index = final_fitnesses.index(max(final_fitnesses))
    best_individual = population[best_index]
    return best_individual



# --------------------------- Main Execution --------------------------- #

def main():
    # Step 1: Load MIDI files
    print("Loading MIDI files...")
    # For initial testing, limit the number of files to speed up loading
    midi_dataset = load_midi_files_parallel(DATASET_PATH, max_files=MAX_INITIAL_FILES, log_file_path=LOG_FILE_PATH)
    if not midi_dataset:
        print("No MIDI files loaded. Exiting.")
        return

    # Step 2: Run Evolutionary Algorithm
    print("Starting evolutionary music generation...")
    best_music = evolutionary_music_generation(
        midi_dataset=midi_dataset,
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        selection_size=SELECTION_SIZE,
        mutation_rate=MUTATION_RATE
    )

    # Step 3: Convert the best individual back to MIDI and save
    print("Converting the best individual to MIDI...")
    output_midi = note_sequence_to_multi_track_midi(best_music)
    output_midi.write(OUTPUT_MIDI_PATH)
    print(f"Evolved music saved as '{OUTPUT_MIDI_PATH}'.")

if __name__ == "__main__":
    main()