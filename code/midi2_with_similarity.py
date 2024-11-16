import os
import pretty_midi
import random
import copy
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

# --------------------------- Configuration --------------------------- #

# Path to your dataset
DATASET_PATH = 'Users\ivanv\.cache\kagglehub\datasets\imsparsh\lakh-midi-clean\versions\1'

# Evolutionary Algorithm Parameters
POPULATION_SIZE = 20        # Number of individuals in the population
SELECTION_SIZE = 10         # Number of top individuals to select
GENERATIONS = 10           # Number of generations to evolve
MUTATION_RATE = 0.6         # Probability of mutation for each individual
OUTPUT_MIDI_PATH = 'evolved_music.mid'  # Path to save the evolved MIDI file
LOG_FILE_PATH = 'midi_load_log.txt'     # Path to save the MIDI file load log

# Mutation Parameters
PITCH_SHIFT_RANGE = (-2, 2)   # Semitones for pitch shifting
TIME_STRETCH_FACTOR = (0.9, 1.1)  # Factor range for time stretching
MAX_PITCH = 127
MAX_TIME = 100
VELOCITY_RANGE = (30, 100)

MAX_SIMULTANEOUS_NOTES = 4  # Maximum notes that can play at once
VELOCITY_RANGE = (40, 90)   # Reduced velocity range for quieter notes

# Maximum number of MIDI files to load initially (for testing)
MAX_INITIAL_FILES = 100
TEMPO = random.uniform(100, 150)

# Add new constants for smoother music generation
LEGATO_FACTOR = 0.1  # Overlap between consecutive notes
MIN_NOTE_DURATION = 0.1  # Minimum duration for any note
MAX_NOTE_DURATION = 2.0  # Maximum duration for any note
VELOCITY_SMOOTHING = 0.3  # Factor for smoothing velocity changes
PHRASE_LENGTH = 4  # Typical length of a musical phrase in beats

MAX_SCORE = 100.0  # Maximum possible score 

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
    Now includes chords by gathering multiple notes that start at similar times.
    """
    # Group notes by similar start times to capture chords
    time_threshold = 0.05  # Notes starting within 50ms will be considered part of the same chord
    notes_by_time = {}
    
    for note in instrument.notes:
        # Round start time to nearest threshold to group chord notes
        chord_time = round(note.start / time_threshold) * time_threshold
        if chord_time not in notes_by_time:
            notes_by_time[chord_time] = []
        notes_by_time[chord_time].append({
            'pitch': note.pitch,
            'start': note.start,
            'end': note.end,
            'velocity': note.velocity
        })
    
    # Flatten the grouped notes while maintaining their order
    notes = []
    for start_time in sorted(notes_by_time.keys()):
        notes.extend(notes_by_time[start_time])
    
    return notes

def smooth_velocities(note_sequence: NoteSequence) -> NoteSequence:
    """
    Smooth out velocity transitions between consecutive notes.
    """
    if len(note_sequence) < 2:
        return note_sequence
    
    smoothed = copy.deepcopy(note_sequence)
    
    # Sort by start time
    smoothed.sort(key=lambda x: x['start'])
    
    # Apply moving average to velocities
    window_size = 3
    for i in range(1, len(smoothed) - 1):
        prev_vel = smoothed[i-1]['velocity']
        current_vel = smoothed[i]['velocity']
        next_vel = smoothed[i+1]['velocity']
        
        # Weighted average favoring the current velocity
        smoothed[i]['velocity'] = int(
            0.2 * prev_vel +
            0.6 * current_vel +
            0.2 * next_vel
        )
    
    return smoothed

def apply_legato(note_sequence: NoteSequence) -> NoteSequence:
    """
    Create smoother transitions between notes by extending note durations.
    """
    if len(note_sequence) < 2:
        return note_sequence
    
    legato = copy.deepcopy(note_sequence)
    legato.sort(key=lambda x: x['start'])
    
    for i in range(len(legato) - 1):
        current_note = legato[i]
        next_note = legato[i + 1]
        
        # Extend current note to overlap with next note
        if next_note['start'] > current_note['start']:
            overlap = min(
                LEGATO_FACTOR,
                (next_note['start'] - current_note['start']) * 0.5
            )
            current_note['end'] = next_note['start'] + overlap
    
    return legato

def create_phrase_structure(note_sequence: NoteSequence) -> NoteSequence:
    """
    Organize notes into musical phrases with natural breathing points.
    """
    if len(note_sequence) < 4:
        return note_sequence
    
    structured = copy.deepcopy(note_sequence)
    structured.sort(key=lambda x: x['start'])
    
    # Group notes into phrases
    phrase_length = PHRASE_LENGTH  # in beats
    current_phrase_start = structured[0]['start']
    
    for i in range(len(structured)):
        # Add slight pause between phrases
        if structured[i]['start'] - current_phrase_start >= phrase_length:
            # Add a small gap after the previous note
            if i > 0:
                structured[i-1]['end'] = min(
                    structured[i-1]['end'],
                    structured[i]['start'] - 0.1
                )
            current_phrase_start = structured[i]['start']
    
    return structured

def add_dynamic_expression(note_sequence: NoteSequence) -> NoteSequence:
    """
    Add natural dynamic expression to the music.
    """
    if not note_sequence:
        return note_sequence
    
    expressive = copy.deepcopy(note_sequence)
    expressive.sort(key=lambda x: x['start'])
    
    # Create natural crescendos and diminuendos
    phrase_length = PHRASE_LENGTH
    for i in range(len(expressive)):
        position_in_phrase = (expressive[i]['start'] % phrase_length) / phrase_length
        
        # Create an arch-shaped dynamic curve
        dynamic_factor = 1.0 + 0.2 * np.sin(position_in_phrase * np.pi)
        expressive[i]['velocity'] = int(
            min(127, expressive[i]['velocity'] * dynamic_factor)
        )
    
    return expressive

# Modify the existing note_sequence_to_midi function
def note_sequence_to_midi(note_sequence: NoteSequence, tempo: float = TEMPO) -> pretty_midi.PrettyMIDI:
    """
    Convert a note sequence to MIDI with enhanced musical expression.
    """
    # Apply musical enhancements
    enhanced_sequence = note_sequence
    enhanced_sequence = smooth_velocities(enhanced_sequence)
    enhanced_sequence = apply_legato(enhanced_sequence)
    enhanced_sequence = create_phrase_structure(enhanced_sequence)
    enhanced_sequence = add_dynamic_expression(enhanced_sequence)
    
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Group notes by start time
    notes_by_time = {}
    for note in enhanced_sequence:
        start_time = round(note['start'], 2)
        if start_time not in notes_by_time:
            notes_by_time[start_time] = []
        notes_by_time[start_time].append(note)
    
    # Create instruments with different timbres
    melody = pretty_midi.Instrument(program=0)  # Piano
    harmony = pretty_midi.Instrument(program=48)  # String ensemble
    
    for start_time, notes in notes_by_time.items():
        # Limit simultaneous notes
        notes = sorted(notes, key=lambda x: x['velocity'], reverse=True)[:MAX_SIMULTANEOUS_NOTES]
        
        if len(notes) == 1:
            # Single note goes to melody
            note = notes[0]
            midi_note = pretty_midi.Note(
                velocity=int(min(127, max(40, note['velocity']))),
                pitch=int(min(127, max(0, note['pitch']))),
                start=float(note['start']),
                end=float(note['end'])
            )
            melody.notes.append(midi_note)
        else:
            # Distribute notes between melody and harmony
            main_note = max(notes, key=lambda x: x['pitch'])
            midi_note = pretty_midi.Note(
                velocity=int(min(127, max(40, main_note['velocity']))),
                pitch=int(min(127, max(0, main_note['pitch']))),
                start=float(main_note['start']),
                end=float(main_note['end'])
            )
            melody.notes.append(midi_note)
            
            # Other notes to harmony with smoother volume
            for note in notes:
                if note != main_note:
                    midi_note = pretty_midi.Note(
                        velocity=int(min(127, max(30, note['velocity'] * 0.8))),
                        pitch=int(min(127, max(0, note['pitch']))),
                        start=float(note['start']),
                        end=float(note['end'])
                    )
                    harmony.notes.append(midi_note)
    
    # Add instruments to MIDI file
    if melody.notes:
        midi.instruments.append(melody)
    if harmony.notes:
        midi.instruments.append(harmony)
    
    return midi

# --------------------------- Mutation Operators --------------------------- #
def add_chord(note_sequence: NoteSequence) -> NoteSequence:
    """
    Add a chord based on music theory principles, with controlled timing.
    """
    if not note_sequence:
        return note_sequence
    
    # Choose a random note as the root, preferring notes with longer duration
    valid_notes = [note for note in note_sequence 
                  if (note['end'] - note['start']) > 0.2]  # Only use notes that are longer than 0.2 seconds
    if not valid_notes:
        return note_sequence
    
    root_note = random.choice(valid_notes)
    
    # Define simpler chord structures (mainly triads)
    chord_types = [
        [0, 4, 7],      # Major triad
        [0, 3, 7],      # Minor triad
    ]
    
    # Choose a random chord type
    intervals = random.choice(chord_types)
    
    # Add chord notes with the same duration as the root note
    new_notes = []
    for interval in intervals[1:]:  # Skip root as it already exists
        chord_note = copy.deepcopy(root_note)
        chord_note['pitch'] = root_note['pitch'] + interval
        # Ensure pitch is in MIDI range
        while chord_note['pitch'] > 127:
            chord_note['pitch'] -= 12
        if 0 <= chord_note['pitch'] <= 127:
            # Slightly reduce velocity of harmony notes
            chord_note['velocity'] = int(chord_note['velocity'] * 0.8)
            new_notes.append(chord_note)
    
    note_sequence.extend(new_notes)
    note_sequence.sort(key=lambda x: x['start'])
    
    return note_sequence


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
    Enhanced mutation function with more musical operations.
    """
    mutation_type = random.choice([
        'pitch_shift', 'time_stretch', 'add_note', 'delete_note', 
        'modify_note', 'add_chord', 'add_harmony'
    ])
    
    if mutation_type == 'add_chord':
        return add_chord(note_sequence)
    elif mutation_type == 'add_harmony':
        # Add parallel harmony (thirds or fifths above existing notes)
        if note_sequence:
            harmony_interval = random.choice([4, 7])  # Major third or perfect fifth
            new_sequence = copy.deepcopy(note_sequence)
            for note in new_sequence:
                harmony_note = copy.deepcopy(note)
                harmony_note['pitch'] += harmony_interval
                if harmony_note['pitch'] <= 127:
                    note_sequence.append(harmony_note)
        return note_sequence
    else:
        # Original mutation types
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
    
    return note_sequence


# --------------------------- Enhanced Fitness Function --------------------------- #

def fitness(note_sequence: NoteSequence) -> float:
    """
    Enhanced fitness function with normalized scores to prevent infinite values.
    """
    if not note_sequence:
        return 0.0

    WEIGHTS = {
        'unique_pitches': 0.3,
        'total_notes': 0.2,
        'smoothness': 0.3,
        'phrase_structure': 0.2
    }

    
    try:
        # Calculate normalized unique pitches score (0-1)
        unique_pitches = len(set(note['pitch'] for note in note_sequence))
        unique_pitches_score = min(1.0, unique_pitches / 50.0)  # Normalize to max of 50 unique pitches
        
        # Calculate normalized total notes score (0-1)
        total_notes = len(note_sequence)
        total_notes_score = min(1.0, total_notes / 500.0)  # Normalize to max of 500 notes
        
        # Calculate smoothness score (0-1)
        smoothness_score = 0.0
        if len(note_sequence) > 1:
            sorted_notes = sorted(note_sequence, key=lambda x: x['start'])
            total_transitions = len(sorted_notes) - 1
            smooth_transitions = 0
            
            for i in range(total_transitions):
                current_note = sorted_notes[i]
                next_note = sorted_notes[i + 1]
                
                # Calculate pitch difference
                pitch_diff = abs(current_note['pitch'] - next_note['pitch'])
                time_diff = max(0.0001, next_note['start'] - current_note['end'])
                
                # Award points for smooth transitions
                if pitch_diff <= 12 and time_diff < 0.5:  # Within an octave and close in time
                    smooth_transitions += 1
            
            smoothness_score = smooth_transitions / total_transitions if total_transitions > 0 else 0.0
        
        # Calculate phrase structure score (0-1)
        phrase_score = 0.0
        if len(note_sequence) > 1:
            sorted_notes = sorted(note_sequence, key=lambda x: x['start'])
            total_possible_phrases = max(1, len(sorted_notes) // 8)  # Assume ideal phrase length of 8 notes
            phrase_boundaries = 0
            
            for i in range(1, len(sorted_notes)):
                time_diff = sorted_notes[i]['start'] - sorted_notes[i-1]['end']
                if 0.1 <= time_diff <= 0.5:  # Reasonable phrase boundary
                    phrase_boundaries += 1
            
            phrase_score = min(1.0, phrase_boundaries / total_possible_phrases)
        
        # Calculate weighted sum of all components
        final_score = (
            WEIGHTS['unique_pitches'] * unique_pitches_score +
            WEIGHTS['total_notes'] * total_notes_score +
            WEIGHTS['smoothness'] * smoothness_score +
            WEIGHTS['phrase_structure'] * phrase_score
        ) * MAX_SCORE
        
        return max(0.0, min(MAX_SCORE, final_score))  # Ensure score is between 0 and MAX_SCORE
        
    except Exception as e:
        print(f"Error in fitness calculation: {e}")
        return 0.0  # Return 0 if any calculation fails

# --------------------------- Evolutionary Algorithm --------------------------- #

def initialize_population(midi_dataset: List[pretty_midi.PrettyMIDI], population_size: int, max_initial_files: int = 500) -> List[NoteSequence]:
    """
    Initialize the population with more complex note sequences from multiple instruments.
    """
    population = []
    limited_dataset = midi_dataset[:max_initial_files]
    
    for _ in range(population_size):
        midi = random.choice(limited_dataset)
        combined_notes = []
        
        # Combine notes from all instruments
        for instrument in midi.instruments:
            if instrument.notes:
                notes = midi_to_note_sequence(instrument)
                combined_notes.extend(notes)
        
        # Sort by start time
        if combined_notes:
            combined_notes.sort(key=lambda x: x['start'])
            # Limit length if needed
            if len(combined_notes) > 1000:
                combined_notes = combined_notes[:1000]
            population.append(combined_notes)
    
    return population



def select_best(population: List[NoteSequence], fitnesses: List[float], selection_size: int) -> List[NoteSequence]:
    """
    Select the top-performing individuals based on fitness scores, with safety checks.
    """
    if not population or not fitnesses:
        return []
    
    # Remove any infinite or NaN values
    valid_pairs = [(p, f) for p, f in zip(population, fitnesses) if f != float('inf') and not np.isnan(f)]
    
    if not valid_pairs:
        return random.sample(population, min(selection_size, len(population)))
    
    # Sort based on valid fitness scores
    paired_sorted = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
    
    # Select the top individuals
    selected = [individual for individual, score in paired_sorted[:selection_size]]
    
    # If we don't have enough selected individuals, add random ones
    while len(selected) < selection_size and len(population) > len(selected):
        remaining = [p for p in population if p not in selected]
        selected.append(random.choice(remaining))
    
    return selected

def create_next_generation(selected: List[NoteSequence], population_size: int, mutation_rate: float) -> List[NoteSequence]:
    """
    Create the next generation with safety checks.
    """
    if not selected:
        return []
    
    next_generation = []
    max_attempts = population_size * 2  # Prevent infinite loops
    attempts = 0
    
    while len(next_generation) < population_size and attempts < max_attempts:
        parent = random.choice(selected)
        try:
            child = copy.deepcopy(parent)
            if random.random() < mutation_rate:
                child = mutate(child)
            if child:  # Only add valid children
                next_generation.append(child)
        except Exception as e:
            print(f"Error in creating child: {e}")
        attempts += 1
    
    # If we couldn't create enough valid children, fill with copies of selected individuals
    while len(next_generation) < population_size and selected:
        next_generation.append(copy.deepcopy(random.choice(selected)))
    
    return next_generation

def evolutionary_music_generation(midi_dataset: List[pretty_midi.PrettyMIDI],
                                generations: int,
                                population_size: int,
                                selection_size: int,
                                mutation_rate: float) -> NoteSequence:
    """
    Run the evolutionary algorithm with additional error handling.
    """
    try:
        population = initialize_population(midi_dataset, population_size)
        if not population:
            print("Failed to initialize population")
            return []
        
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(1, generations + 1):
            try:
                # Evaluate fitness with safety checks
                fitnesses = []
                for individual in population:
                    try:
                        fit = fitness(individual)
                        if fit == float('inf') or np.isnan(fit):
                            fit = 0.0
                        fitnesses.append(fit)
                    except Exception as e:
                        print(f"Error calculating fitness: {e}")
                        fitnesses.append(0.0)
                
                current_best = max(fitnesses)
                current_avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                
                print(f"Generation {generation}: Best Fitness = {current_best:.2f}, Avg Fitness = {current_avg:.2f}")
                
                # Update best individual if we found a better one
                if current_best > best_fitness:
                    best_fitness = current_best
                    best_individual = population[fitnesses.index(current_best)]
                
                # Selection
                selected = select_best(population, fitnesses, selection_size)
                if not selected:
                    print("Selection failed, restarting population")
                    population = initialize_population(midi_dataset, population_size)
                    continue
                
                # Create next generation
                population = create_next_generation(selected, population_size, mutation_rate)
                if not population:
                    print("Failed to create next generation, restarting population")
                    population = initialize_population(midi_dataset, population_size)
                
            except Exception as e:
                print(f"Error in generation {generation}: {e}")
                population = initialize_population(midi_dataset, population_size)
        
        return best_individual if best_individual else population[0] if population else []
    
    except Exception as e:
        print(f"Fatal error in evolutionary music generation: {e}")
        return []

# --------------------------- Similarity Calculation --------------------------- #

def extract_midi_features(midi_data: pretty_midi.PrettyMIDI) -> Dict:
    """
    Extracts enhanced musical features from a MIDI file.
    """
    features = {
        "notes": set(),
        "note_durations": [],
        "note_onsets": [],
        "instruments": set(),
        "velocity_profile": [],
        "pitch_intervals": [],
        "rhythm_patterns": [],
        "chord_progressions": [],
        "key_signature": None,
        "tempo_changes": []
    }
    
    if not midi_data.instruments:
        return features
        
    # Extract basic features
    for instrument in midi_data.instruments:
        features["instruments"].add(instrument.program)
        notes = sorted(instrument.notes, key=lambda x: x.start)
        
        for i, note in enumerate(notes):
            if note.end > note.start:
                features["notes"].add(note.pitch)
                features["note_durations"].append(note.end - note.start)
                features["note_onsets"].append(note.start)
                features["velocity_profile"].append(note.velocity)
                
                # Calculate pitch intervals between consecutive notes
                if i > 0:
                    interval = note.pitch - notes[i-1].pitch
                    features["pitch_intervals"].append(interval)
                
                # Extract rhythm patterns (time between note onsets)
                if i > 0:
                    rhythm = note.start - notes[i-1].start
                    features["rhythm_patterns"].append(rhythm)
    
    # Extract tempo changes
    features["tempo_changes"] = midi_data.get_tempo_changes()
    
    # Estimate key signature
    features["key_signature"] = midi_data.estimate_key()
    
    # Extract chord progressions using music21 if available
    try:
        import music21
        score = music21.converter.parse(midi_data)
        chords = score.chordify()
        features["chord_progressions"] = [str(c) for c in chords.recurse().getElementsByClass('Chord')]
    except:
        pass
    
    # Convert to numpy arrays for efficient computation
    for key in ["note_durations", "note_onsets", "velocity_profile", 
                "pitch_intervals", "rhythm_patterns"]:
        if features[key]:
            features[key] = np.array(features[key], dtype=np.float32)
    
    return features

def compare_features(final_features: Dict, parent_features: Dict, weights: Dict = None) -> float:
    """
    Enhanced comparison of musical features between two MIDI files.
    """
    if weights is None:
        weights = {
            "notes": 0.2,
            "durations": 0.15,
            "onsets": 0.1,
            "instruments": 0.1,
            "velocity": 0.1,
            "intervals": 0.15,
            "rhythm": 0.1,
            "key": 0.05,
            "chords": 0.05
        }
    
    similarity = {}
    
    # Basic feature comparisons
    all_notes = final_features["notes"].union(parent_features["notes"])
    common_notes = final_features["notes"].intersection(parent_features["notes"])
    similarity["notes"] = len(common_notes) / len(all_notes) if all_notes else 0
    
    # Compare note durations distribution
    if len(final_features["note_durations"]) and len(parent_features["note_durations"]):
        duration_hist1 = np.histogram(final_features["note_durations"], bins=20)[0]
        duration_hist2 = np.histogram(parent_features["note_durations"], bins=20)[0]
        similarity["durations"] = 1 - np.mean(np.abs(duration_hist1 - duration_hist2))
    else:
        similarity["durations"] = 0
    
    # Compare velocity profiles
    if len(final_features["velocity_profile"]) and len(parent_features["velocity_profile"]):
        vel_hist1 = np.histogram(final_features["velocity_profile"], bins=20)[0]
        vel_hist2 = np.histogram(parent_features["velocity_profile"], bins=20)[0]
        similarity["velocity"] = 1 - np.mean(np.abs(vel_hist1 - vel_hist2))
    else:
        similarity["velocity"] = 0
    
    # Compare pitch intervals (melodic contour)
    if len(final_features["pitch_intervals"]) and len(parent_features["pitch_intervals"]):
        int_hist1 = np.histogram(final_features["pitch_intervals"], bins=24)[0]  # 2 octaves
        int_hist2 = np.histogram(parent_features["pitch_intervals"], bins=24)[0]
        similarity["intervals"] = 1 - np.mean(np.abs(int_hist1 - int_hist2))
    else:
        similarity["intervals"] = 0
    
    # Compare rhythm patterns
    if len(final_features["rhythm_patterns"]) and len(parent_features["rhythm_patterns"]):
        rhythm_hist1 = np.histogram(final_features["rhythm_patterns"], bins=10)[0]
        rhythm_hist2 = np.histogram(parent_features["rhythm_patterns"], bins=10)[0]
        similarity["rhythm"] = 1 - np.mean(np.abs(rhythm_hist1 - rhythm_hist2))
    else:
        similarity["rhythm"] = 0
    
    # Compare key signatures
    if final_features["key_signature"] and parent_features["key_signature"]:
        key_distance = abs(final_features["key_signature"].tonic.midi - 
                         parent_features["key_signature"].tonic.midi)
        similarity["key"] = 1 - (key_distance / 12)  # Normalize by octave
    else:
        similarity["key"] = 0
    
    # Compare chord progressions
    if final_features["chord_progressions"] and parent_features["chord_progressions"]:
        common_chords = set(final_features["chord_progressions"]).intersection(
            set(parent_features["chord_progressions"]))
        all_chords = set(final_features["chord_progressions"]).union(
            set(parent_features["chord_progressions"]))
        similarity["chords"] = len(common_chords) / len(all_chords) if all_chords else 0
    else:
        similarity["chords"] = 0
    
    # Calculate weighted average
    total_similarity = sum(similarity[key] * weights[key] for key in weights if key in similarity)
    return total_similarity * 100

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
