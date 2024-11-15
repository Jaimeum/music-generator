import os
import pretty_midi
import random
import copy
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --------------------------- Configuration --------------------------- #

# Path to your dataset
DATASET_PATH = '/Users/jaimeuria/.cache/kagglehub/datasets/imsparsh/lakh-midi-clean/versions/1'

# Evolutionary Algorithm Parameters
POPULATION_SIZE = 50        # Number of individuals in the population
SELECTION_SIZE = 10         # Number of top individuals to select
GENERATIONS = 100           # Number of generations to evolve
MUTATION_RATE = 0.8         # Probability of mutation for each individual
OUTPUT_MIDI_PATH = 'evolved_music.mid'  # Path to save the evolved MIDI file
LOG_FILE_PATH = 'midi_load_log.txt'     # Path to save the MIDI file load log

# Mutation Parameters
PITCH_SHIFT_RANGE = (-2, 2)   # Semitones for pitch shifting
TIME_STRETCH_FACTOR = (0.9, 1.1)  # Factor range for time stretching
MAX_PITCH = 127
MAX_TIME = 100
VELOCITY_RANGE = (30, 100)

# Maximum number of MIDI files to load initially (for testing)
MAX_INITIAL_FILES = 500

# --------------------------- Data Structures --------------------------- #

# Define the structure for a musical note
Note = Dict[str, Any]
NoteSequence = List[Note]

# --------------------------- Helper Functions --------------------------- #

def load_single_midi(midi_path: str) -> pretty_midi.PrettyMIDI:
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

def load_midi_files_parallel(dataset_path: str, max_files: int = None, log_file_path: str = None) -> List[pretty_midi.PrettyMIDI]:
    """
    Load MIDI files using parallel processing.
    Returns a list of successfully loaded PrettyMIDI objects.
    Logs and skips files that cause errors.
    Optionally limits the number of MIDI files loaded.
    Writes the file paths of the loaded MIDI files to a log file.
    """
    midi_files = []
    error_files = []
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

    # Use multiprocessing Pool to load files in parallel
    with Pool(processes=cpu_count()) as pool:
        for midi in tqdm(pool.imap_unordered(load_single_midi, all_midi_paths), total=total_files, desc="Loading MIDI Files"):
            if midi is not None:
                midi_files.append(midi)
                if log_file_path:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"{all_midi_paths[len(midi_files) - 1]}\n")
            else:
                # Errors are already printed in load_single_midi
                pass

    print(f"Successfully loaded {len(midi_files)} MIDI files.")
    return midi_files

def midi_to_note_sequence(midi: pretty_midi.PrettyMIDI) -> NoteSequence:
    """
    Convert a PrettyMIDI object to a note sequence.
    """
    notes = []
    for instrument in midi.instruments:
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

def note_sequence_to_midi(note_sequence: NoteSequence, tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
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

def pitch_shift(note_sequence: NoteSequence, semitones: int) -> NoteSequence:
    """
    Transpose all notes in the sequence by a given number of semitones.
    """
    shifted = copy.deepcopy(note_sequence)
    for note in shifted:
        note['pitch'] += semitones
        # Ensure pitch stays within MIDI range
        note['pitch'] = max(0, min(MAX_PITCH, note['pitch']))
    return shifted

def time_stretch(note_sequence: NoteSequence, factor: float) -> NoteSequence:
    """
    Stretch the timing of all notes by a given factor.
    """
    stretched = copy.deepcopy(note_sequence)
    for note in stretched:
        note['start'] *= factor
        note['end'] *= factor
    return stretched

def add_random_note(note_sequence: NoteSequence) -> NoteSequence:
    """
    Add a random note to the sequence.
    """
    new_note = {
        'pitch': random.randint(0, MAX_PITCH),
        'start': random.uniform(0, MAX_TIME),
        'end': random.uniform(0, MAX_TIME),
        'velocity': random.randint(VELOCITY_RANGE[0], VELOCITY_RANGE[1])
    }
    # Ensure end is after start
    if new_note['end'] <= new_note['start']:
        new_note['end'] = new_note['start'] + random.uniform(0.1, 1.0)
    note_sequence.append(new_note)
    # Sort by start time
    note_sequence.sort(key=lambda x: x['start'])
    return note_sequence

def delete_random_note(note_sequence: NoteSequence) -> NoteSequence:
    """
    Delete a random note from the sequence.
    """
    if not note_sequence:
        return note_sequence
    idx = random.randint(0, len(note_sequence) - 1)
    del note_sequence[idx]
    return note_sequence

def modify_random_note(note_sequence: NoteSequence) -> NoteSequence:
    """
    Modify a random attribute of a random note in the sequence.
    """
    if not note_sequence:
        return note_sequence
    idx = random.randint(0, len(note_sequence) - 1)
    note = note_sequence[idx]
    modification = random.choice(['pitch', 'velocity', 'start', 'end'])
    if modification == 'pitch':
        note['pitch'] += random.choice([-1, 1])
        note['pitch'] = max(0, min(MAX_PITCH, note['pitch']))
    elif modification == 'velocity':
        note['velocity'] += random.choice([-5, 5])
        note['velocity'] = max(0, min(127, note['velocity']))
    elif modification == 'start':
        note['start'] += random.uniform(-0.1, 0.1)
        note['start'] = max(0, note['start'])
    elif modification == 'end':
        note['end'] += random.uniform(-0.1, 0.1)
        note['end'] = max(note['start'] + 0.1, note['end'])
    return note_sequence

def mutate(note_sequence: NoteSequence) -> NoteSequence:
    """
    Apply a random mutation to the note sequence.
    """
    mutation_type = random.choice(['pitch_shift', 'time_stretch', 'add_note', 'delete_note', 'modify_note'])
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

def fitness(note_sequence: NoteSequence) -> float:
    """
    Calculate the fitness of a note sequence.
    Higher fitness indicates a "better" musical piece.
    """
    if not note_sequence:
        return 0.0
    unique_pitches = len(set(note['pitch'] for note in note_sequence))
    total_notes = len(note_sequence)
    total_duration = sum(note['end'] - note['start'] for note in note_sequence)
    
    # Normalize unique pitches and total notes
    unique_pitches_score = unique_pitches
    total_notes_score = total_notes / 10  # Adjust divisor to scale appropriately
    
    # Example: Combine the two scores
    fitness_score = unique_pitches_score + total_notes_score
    
    # Additional metrics can be added here for a more sophisticated fitness function
    
    return fitness_score

# --------------------------- Evolutionary Algorithm --------------------------- #

def initialize_population(midi_dataset: List[pretty_midi.PrettyMIDI], population_size: int, max_initial_files: int = 500) -> List[NoteSequence]:
    """
    Initialize the population with random note sequences from the dataset.
    Optionally limit the number of MIDI files to load.
    """
    population = []
    limited_dataset = midi_dataset[:max_initial_files]  # Limit the dataset
    for _ in range(population_size):
        midi = random.choice(limited_dataset)
        note_sequence = midi_to_note_sequence(midi)
        # Optionally limit the number of notes to manage complexity
        if len(note_sequence) > 1000:
            note_sequence = note_sequence[:1000]
        population.append(note_sequence)
    return population

def select_best(population: List[NoteSequence], fitnesses: List[float], selection_size: int) -> List[NoteSequence]:
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

def create_next_generation(selected: List[NoteSequence], population_size: int, mutation_rate: float) -> List[NoteSequence]:
    """
    Create the next generation by mutating selected individuals.
    """
    next_generation = []
    while len(next_generation) < population_size:
        parent = random.choice(selected)
        child = copy.deepcopy(parent)
        if random.random() < mutation_rate:
            child = mutate(child)
        next_generation.append(child)
    return next_generation

def evolutionary_music_generation(midi_dataset: List[pretty_midi.PrettyMIDI],
                                  generations: int,
                                  population_size: int,
                                  selection_size: int,
                                  mutation_rate: float) -> NoteSequence:
    """
    Run the evolutionary algorithm to generate music.
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
        
        # Create next generation
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
    output_midi = note_sequence_to_midi(best_music)
    output_midi.write(OUTPUT_MIDI_PATH)
    print(f"Evolved music saved as '{OUTPUT_MIDI_PATH}'.")

if __name__ == "__main__":
    main()