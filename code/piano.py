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

# --------------------------- Configuration --------------------------- #

# Path to your dataset
DATASET_PATH = '/Users/jaimeuria/ITAM/Databases/GiantMIDI-Piano/surname_checked_midis'

# Evolutionary Algorithm Parameters
POPULATION_SIZE = 100        # Number of individuals in the population
SELECTION_SIZE = 50         # Number of top individuals to select
GENERATIONS = 100            # Number of generations to evolve
MUTATION_RATE = 0.5          # Probability of mutation for each individual
OUTPUT_MIDI_PATH = 'segundo.mid'  # Path to save the evolved MIDI file
LOG_FILE_PATH = 'midi_load_log.txt'     # Path to save the MIDI file load log

# Mutation Parameters
PITCH_SHIFT_RANGE = (-2, 2)   # Semitones for pitch shifting
TIME_STRETCH_FACTOR = (0.9, 1.1)  # Factor range for time stretching
MIN_PITCH = 45                # Minimum acceptable MIDI pitch (A0)
MAX_PITCH = 85               # Maximum acceptable MIDI pitch (C8)
MAX_TIME = 100
MAX_PITCH_JUMP = 12          # One octave

MAX_SIMULTANEOUS_NOTES = 6  # Maximum notes that can play at once
VELOCITY_RANGE = (60, 100)   # Adjusted velocity range for clearer dynamics

# Rhythm Configuration
TEMPO = 100.0                # Beats per minute
BEAT_DIVISION = 4            # Number of divisions per beat (e.g., 4 for sixteenth notes)

# Derived Constants
SECONDS_PER_BEAT = 60.0 / TEMPO
BEAT_DURATION = SECONDS_PER_BEAT / BEAT_DIVISION  # Duration of each beat division

# Maximum number of MIDI files to load initially (for testing)
MAX_INITIAL_FILES = 200

# Add new constants for smoother music generation
LEGATO_FACTOR = 0.6  # Overlap between consecutive notes
MIN_NOTE_DURATION = BEAT_DURATION  # Minimum duration for any note aligned to grid
MAX_NOTE_DURATION = 2.0  # Maximum duration for any note
VELOCITY_SMOOTHING = 0.5  # Factor for smoothing velocity changes
PHRASE_LENGTH = 5  # Typical length of a musical phrase in beats
IDEAL_PHRASE_LENGTH = 8  # Ideal number of notes per phrase
PREFERRED_VELOCITY = 80  # Preferred mean velocity
PREFERRED_VELOCITY_MIN = 60  # Lower bound for preferred mean velocity
PREFERRED_VELOCITY_MAX = 100  # Upper bound for preferred mean velocity

MAX_SCORE = 1000.0  # Maximum possible score 

# --------------------------- Data Structures --------------------------- #

# Define the structure for a musical note
Note = Dict[str, Any]
NoteSequence = List[Note]

# --------------------------- Helper Functions --------------------------- #

# Define major and minor scales for reference
MAJOR_SCALES = {key: [(key + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]] for key in range(12)}
MINOR_SCALES = {key: [(key + interval) % 12 for interval in [0, 2, 3, 5, 7, 8, 10]] for key in range(12)}

def determine_key(sorted_notes):
    pitch_classes = [note['pitch'] % 12 for note in sorted_notes]
    pitch_counter = Counter(pitch_classes)
    most_common = pitch_counter.most_common()
    if not most_common:
        return None, None

    # Find the key with the highest match to major and minor scales
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
    Adjust the velocities to maintain an average around the target_mean.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        target_mean (float): Desired mean velocity.

    Returns:
        NoteSequence: Velocity-adjusted note sequence.
    """
    if not note_sequence:
        return note_sequence

    current_mean = np.mean([note['velocity'] for note in note_sequence])
    adjustment_factor = target_mean / current_mean if current_mean != 0 else 1.0

    adjusted = copy.deepcopy(note_sequence)
    for note in adjusted:
        new_velocity = int(note['velocity'] * adjustment_factor)
        # Clamp velocity within the defined range
        note['velocity'] = max(VELOCITY_RANGE[0], min(VELOCITY_RANGE[1], new_velocity))

    return adjusted


def apply_swing(note_sequence: NoteSequence, swing_factor: float = 0.2) -> NoteSequence:
    """
    Apply a swing feel to the note sequence by delaying off-beat notes.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        swing_factor (float): Fraction of the beat to delay (e.g., 0.2 for 20% delay).

    Returns:
        NoteSequence: The swing-applied note sequence.
    """
    swung = copy.deepcopy(note_sequence)
    swung.sort(key=lambda x: x['start'])

    for note in swung:
        # Determine if the note is on an off-beat subdivision (e.g., eighth notes in 4/4)
        subdivision = note['start'] / BEAT_DURATION
        if abs(subdivision - round(subdivision)) == 0.5:
            # Delay the note by swing_factor of the beat division
            note['start'] += BEAT_DURATION * swing_factor
            note['end'] += BEAT_DURATION * swing_factor
            # Quantize again to maintain alignment
            note['start'] = quantize_time(note['start'])
            note['end'] = quantize_time(note['end'])

    swung.sort(key=lambda x: x['start'])
    return swung


def quantize_time(time: float, division: int = BEAT_DIVISION) -> float:
    """
    Quantize a given time to the nearest beat division.

    Parameters:
        time (float): The original time in seconds.
        division (int): Number of divisions per beat.

    Returns:
        float: Quantized time in seconds.
    """
    return round(time / BEAT_DURATION) * BEAT_DURATION


def validate_note_sequence(note_sequence: NoteSequence) -> NoteSequence:
    """
    Validate and correct a note sequence to ensure all notes are valid and aligned to the beat grid.

    Parameters:
        note_sequence (NoteSequence): The note sequence to validate.

    Returns:
        NoteSequence: Validated and corrected note sequence.
    """
    validated = []
    for note in note_sequence:
        # Ensure end is after start
        if note['end'] <= note['start']:
            note['end'] = note['start'] + MIN_NOTE_DURATION  # Assign a minimal duration aligned to grid
        
        # Quantize start and end times
        note['start'] = quantize_time(note['start'])
        note['end'] = quantize_time(note['end'])
        
        # Remove notes outside the acceptable pitch range
        if MIN_PITCH <= note['pitch'] <= MAX_PITCH:
            # Ensure velocity is within range
            note['velocity'] = max(VELOCITY_RANGE[0], min(127, note['velocity']))
            validated.append(note)
        else:
            # Optionally, log or count the removal of notes
            pass  # Notes outside range are simply removed
    
    # Sort by start time
    validated.sort(key=lambda x: x['start'])
    
    return validated

def get_scale_pitches(root: int, scale_type: str = 'major') -> set:
    """
    Generate a set of pitches based on the root note and scale type.

    Parameters:
        root (int): MIDI pitch number of the root note.
        scale_type (str): Type of scale ('major' or 'minor').

    Returns:
        set: Set of MIDI pitch numbers in the scale.
    """
    if scale_type == 'major':
        intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
    elif scale_type == 'minor':
        intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor scale intervals
    else:
        raise ValueError("Unsupported scale type. Use 'major' or 'minor'.")

    scale_pitches = set(root + interval for interval in intervals)
    return scale_pitches

def mutate_phrases(note_sequence: NoteSequence, phrase_length_beats: int = 8) -> NoteSequence:
    """
    Mutate entire phrases by applying transformations like inversion or retrograde.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        phrase_length_beats (int): Number of beats per phrase.

    Returns:
        NoteSequence: Mutated note sequence with phrase-level transformations.
    """
    if not note_sequence:
        return note_sequence

    mutated = copy.deepcopy(note_sequence)
    mutated.sort(key=lambda x: x['start'])

    # Determine the number of phrases
    phrase_duration = phrase_length_beats * SECONDS_PER_BEAT
    num_phrases = int(math.ceil(max(note['end'] for note in mutated) / phrase_duration))

    for phrase_idx in range(num_phrases):
        # Extract notes in the current phrase
        phrase_notes = [note for note in mutated if phrase_idx * phrase_duration <= note['start'] < (phrase_idx + 1) * phrase_duration]
        if not phrase_notes:
            continue

        # Choose a transformation
        transformation = random.choice(['invert', 'retrograde', 'transpose_up', 'transpose_down'])
        
        if transformation == 'invert':
            # Invert pitches around the first note's pitch
            pivot = phrase_notes[0]['pitch']
            for note in phrase_notes:
                note['pitch'] = pivot - (note['pitch'] - pivot)
        elif transformation == 'retrograde':
            # Reverse the order of notes in time
            phrase_notes_sorted = sorted(phrase_notes, key=lambda x: x['start'], reverse=True)
            for original, retro in zip(phrase_notes, phrase_notes_sorted):
                original['start'], original['end'] = retro['start'], retro['end']
        elif transformation == 'transpose_up':
            # Transpose the entire phrase up by a random interval within the scale
            semitones = random.choice([2, 4, 5, 7, 9, 11])  # Intervals from the scale
            for note in phrase_notes:
                note['pitch'] += semitones
                note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))
        elif transformation == 'transpose_down':
            # Transpose the entire phrase down by a random interval within the scale
            semitones = random.choice([2, 3, 5, 7, 8, 10])  # Intervals from the scale
            for note in phrase_notes:
                note['pitch'] -= semitones
                note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))

    mutated.sort(key=lambda x: x['start'])
    return mutated


def apply_voice_leading(note_sequence: NoteSequence) -> NoteSequence:
    """
    Apply voice leading principles to ensure smooth transitions between notes.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.

    Returns:
        NoteSequence: Note sequence with applied voice leading.
    """
    if len(note_sequence) < 2:
        return note_sequence

    led = copy.deepcopy(note_sequence)
    led.sort(key=lambda x: x['start'])

    for i in range(1, len(led)):
        prev_note = led[i - 1]
        current_note = led[i]

        # Calculate the pitch difference
        pitch_diff = current_note['pitch'] - prev_note['pitch']

        # Limit the pitch leap to a maximum of a fifth (7 semitones)
        if abs(pitch_diff) > 7:
            # Adjust the pitch to reduce the leap
            adjustment = -7 if pitch_diff > 0 else 7
            new_pitch = current_note['pitch'] + adjustment
            # Ensure within MIDI range and scale
            new_pitch = max(MIN_PITCH, min(MAX_PITCH, new_pitch))
            current_note['pitch'] = new_pitch

    return led


def pitch_shift_scale_conscious(note_sequence: NoteSequence, semitones: int, root: int, scale_type: str = 'major') -> NoteSequence:
    """
    Transpose all notes in the sequence by a given number of semitones,
    ensuring they remain within the specified scale.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        semitones (int): Number of semitones to transpose.
        root (int): Root note MIDI pitch number for scale generation.
        scale_type (str): Type of scale ('major' or 'minor').

    Returns:
        NoteSequence: The transposed note sequence.
    """
    scale_pitches = get_scale_pitches(root, scale_type)
    shifted = copy.deepcopy(note_sequence)

    for note in shifted:
        note['pitch'] += semitones
        # Ensure pitch stays within MIDI range
        note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))
        # Adjust pitch to fit within the scale
        if note['pitch'] not in scale_pitches:
            # Find the nearest pitch within the scale
            possible_pitches = [p for p in scale_pitches if MIN_PITCH <= p <= MAX_PITCH]
            if possible_pitches:
                closest_pitch = min(possible_pitches, key=lambda p: abs(p - note['pitch']))
                note['pitch'] = closest_pitch
        # Quantize to maintain rhythm
        note['start'] = quantize_time(note['start'])
        note['end'] = quantize_time(note['end'])
    
    return shifted


def limit_note_density(note_sequence: NoteSequence, max_notes_per_beat: int = 2) -> NoteSequence:
    """
    Limit the number of notes that can start within the same beat division.

    Parameters:
        note_sequence (NoteSequence): The note sequence to limit.
        max_notes_per_beat (int): Maximum number of notes allowed per beat division.

    Returns:
        NoteSequence: Note sequence with limited density.
    """
    density_limited = []
    beat_notes = {}
    
    for note in note_sequence:
        beat = int(note['start'] / BEAT_DURATION)
        if beat not in beat_notes:
            beat_notes[beat] = []
        if len(beat_notes[beat]) < max_notes_per_beat:
            beat_notes[beat].append(note)
            density_limited.append(note)
        else:
            # Optionally, remove or log notes that exceed density
            pass  # Exceeding notes are omitted
    
    return density_limited


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
    Returns a list of successfully loaded PrettyMIDI objects that contain piano instruments.
    Logs and skips files that cause errors or do not contain piano instruments.
    Optionally limits the number of MIDI files loaded.
    Writes the file paths of the loaded MIDI files to a log file.
    """
    midi_files = []
    
    # Since all MIDI files are directly inside dataset_path, no need to list subdirectories
    all_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.mid', '.midi'))]
    total_files = len(all_files)
    print(f"Total MIDI files to load: {total_files}")

    # Apply max_files limit if specified
    if max_files:
        all_files = random.sample(all_files, min(max_files, len(all_files)))
        total_files = len(all_files)
        print(f"Randomly selected {total_files} MIDI files as per max_files parameter.")

    # Construct full paths
    all_midi_paths = [os.path.join(dataset_path, f) for f in all_files]

    # Shuffle the list to randomize file selection
    random.shuffle(all_midi_paths)
    
    # Use multiprocessing Pool to load files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = tqdm(pool.imap_unordered(load_midi_with_path, all_midi_paths), total=total_files, desc="Loading MIDI Files")
        for path, midi in results:
            if midi is not None:
                # Check if the MIDI file contains at least one piano instrument
                has_piano = any(instr.program == 0 or instr.name.lower().find('piano') != -1 for instr in midi.instruments if not instr.is_drum)
                if has_piano:
                    midi_files.append(midi)
                    if log_file_path:
                        # Since there are no subdirectories, log only the filename
                        filename = os.path.basename(path)
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(f"{filename}\n")
                else:
                    print(f"Skipping {path}: No piano instrument found.")
            else:
                # Errors are already printed in load_single_midi
                pass

    print(f"Successfully loaded {len(midi_files)} MIDI files with piano instruments.")
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

def identify_motifs(note_sequence: NoteSequence, motif_length: int = 4) -> List[NoteSequence]:
    """
    Identify recurring motifs within the note sequence.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        motif_length (int): Number of notes defining a motif.

    Returns:
        List[NoteSequence]: List of identified motifs.
    """
    motifs = []
    sequence_length = len(note_sequence)
    for i in range(sequence_length - motif_length + 1):
        candidate = note_sequence[i:i + motif_length]
        # Simple motif identification: exact pitch sequence
        if candidate in motifs:
            continue
        # Check for repetitions
        count = 1
        for j in range(i + motif_length, sequence_length - motif_length + 1):
            if note_sequence[j:j + motif_length] == candidate:
                count += 1
        if count > 1:
            motifs.append(candidate)
    return motifs

def mutate_motifs(note_sequence: NoteSequence, motif_length: int = 4) -> NoteSequence:
    """
    Apply mutations to identified motifs to introduce variations.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        motif_length (int): Number of notes defining a motif.

    Returns:
        NoteSequence: Mutated note sequence with variations in motifs.
    """
    motifs = identify_motifs(note_sequence, motif_length)
    if not motifs:
        return note_sequence

    mutated = copy.deepcopy(note_sequence)

    for motif in motifs:
        # Choose a motif to mutate
        selected_motif = random.choice(motifs)
        # Choose a transformation
        transformation = random.choice(['invert', 'retrograde', 'transpose_up', 'transpose_down'])
        
        # Find all instances of the selected motif
        indices = [i for i in range(len(mutated) - motif_length + 1) if mutated[i:i + motif_length] == selected_motif]
        for idx in indices:
            phrase = mutated[idx:idx + motif_length]
            if transformation == 'invert':
                pivot = phrase[0]['pitch']
                for note in phrase:
                    note['pitch'] = pivot - (note['pitch'] - pivot)
            elif transformation == 'retrograde':
                reversed_phrase = list(reversed(phrase))
                for original, reversed_note in zip(phrase, reversed_phrase):
                    original['pitch'] = reversed_note['pitch']
                    original['velocity'] = reversed_note['velocity']
            elif transformation == 'transpose_up':
                semitones = random.choice([2, 4, 5, 7, 9, 11])  # Intervals from the scale
                for note in phrase:
                    note['pitch'] += semitones
                    note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))
            elif transformation == 'transpose_down':
                semitones = random.choice([2, 3, 5, 7, 8, 10])  # Intervals from the scale
                for note in phrase:
                    note['pitch'] -= semitones
                    note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))

    mutated.sort(key=lambda x: x['start'])
    return mutated


def apply_accentuation(note_sequence: NoteSequence, accent_beats: List[int] = [1, 3]) -> NoteSequence:
    """
    Accentuate notes on specified strong beats.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        accent_beats (List[int]): List of beat numbers to accentuate (1-based).

    Returns:
        NoteSequence: Note sequence with accentuated beats.
    """
    if not note_sequence:
        return note_sequence

    accented = copy.deepcopy(note_sequence)
    accented.sort(key=lambda x: x['start'])

    for note in accented:
        # Determine the beat number within the measure
        beat_number = int((note['start'] % SECONDS_PER_BEAT * BEAT_DIVISION) // BEAT_DURATION) + 1
        if beat_number in accent_beats:
            # Increase velocity for accented notes
            note['velocity'] = min(127, int(note['velocity'] * 1.2))
    
    return accented


def apply_phrase_dynamics(note_sequence: NoteSequence, phrase_length_beats: int = 8) -> NoteSequence:
    """
    Apply dynamic changes to phrases, creating crescendos and diminuendos.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        phrase_length_beats (int): Number of beats per phrase.

    Returns:
        NoteSequence: Note sequence with phrase-level dynamics.
    """
    if not note_sequence:
        return note_sequence

    enhanced = copy.deepcopy(note_sequence)
    enhanced.sort(key=lambda x: x['start'])

    # Calculate phrase boundaries
    phrases = {}
    for note in enhanced:
        phrase_index = int(note['start'] // (phrase_length_beats * SECONDS_PER_BEAT))
        if phrase_index not in phrases:
            phrases[phrase_index] = []
        phrases[phrase_index].append(note)

    for phrase in phrases.values():
        # Determine dynamic trend for the phrase (crescendo or diminuendo)
        trend = random.choice(['crescendo', 'diminuendo'])
        velocities = [note['velocity'] for note in phrase]
        if trend == 'crescendo':
            sorted_velocities = sorted(velocities)
        else:
            sorted_velocities = sorted(velocities, reverse=True)
        
        for i, note in enumerate(phrase):
            note['velocity'] = int(np.interp(i, [0, len(phrase) - 1], [sorted_velocities[0], sorted_velocities[-1]]))
    
    return enhanced


def smooth_velocities(note_sequence: NoteSequence, window_size: int = 3) -> NoteSequence:
    """
    Smooth out velocity transitions between consecutive notes.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        window_size (int): Number of notes to include in the smoothing window.

    Returns:
        NoteSequence: Velocity-smoothed note sequence.
    """
    if len(note_sequence) < 2:
        return note_sequence

    smoothed = copy.deepcopy(note_sequence)
    smoothed.sort(key=lambda x: x['start'])

    for i in range(len(smoothed)):
        window = []
        for j in range(max(0, i - window_size), min(len(smoothed), i + window_size + 1)):
            window.append(smoothed[j]['velocity'])
        smoothed[i]['velocity'] = int(np.mean(window))

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

def note_sequence_to_midi(note_sequence: NoteSequence, tempo: float = TEMPO) -> pretty_midi.PrettyMIDI:
    """
    Convert a note sequence to MIDI with enhanced musical expression.
    Generates a MIDI file containing only a single piano instrument.
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
    
    # Create a single piano instrument
    piano = pretty_midi.Instrument(program=0, name='Acoustic Grand Piano')  # Acoustic Grand Piano
    
    for start_time, notes in notes_by_time.items():
        # Limit simultaneous notes
        notes = sorted(notes, key=lambda x: x['velocity'], reverse=True)[:MAX_SIMULTANEOUS_NOTES]
        
        for note in notes:
            midi_note = pretty_midi.Note(
                velocity=int(min(127, max(40, note['velocity']))),
                pitch=int(min(127, max(0, note['pitch']))),
                start=float(note['start']),
                end=float(note['end'])
            )
            piano.notes.append(midi_note)
    
    # Add the piano instrument to MIDI file
    if piano.notes:
        midi.instruments.append(piano)
    
    return midi



# --------------------------- Mutation Operators --------------------------- #
def add_chord(note_sequence: NoteSequence) -> NoteSequence:
    """
    Add a chord based on music theory principles, aligned with rhythmic patterns.
    Only adds harmonious triads.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.

    Returns:
        NoteSequence: The mutated note sequence with added chord.
    """
    if not note_sequence:
        return note_sequence

    # Choose a random note as the root, preferring notes with longer duration
    valid_notes = [note for note in note_sequence 
                  if (note['end'] - note['start']) >= BEAT_DURATION]  # Use notes at least one beat long
    if not valid_notes:
        return note_sequence

    root_note = random.choice(valid_notes)

    chord_types = [
        # Triads
        [0, 4, 7],          # Major triad
        [0, 3, 7],          # Minor triad
        [0, 3, 6],          # Diminished triad
        [0, 4, 8],          # Augmented triad
        [0, 5, 7],          # Power chord (no third)
        [0, 2, 7],          # Suspended 2nd (sus2)
        [0, 5, 7],          # Suspended 4th (sus4)
        [0, 4, 7, 11],      # Major seventh
        [0, 3, 7, 10],      # Minor seventh
        [0, 4, 7, 10],      # Dominant seventh
        [0, 3, 6, 10],      # Half-diminished seventh
        [0, 3, 6, 9],       # Diminished seventh
        [0, 4, 7, 9],       # Major ninth
        [0, 3, 7, 10, 14],  # Minor thirteenth

        # Extended Chords
        [0, 4, 7, 11, 14],  # Major thirteenth
        [0, 4, 7, 10, 14],  # Dominant thirteenth
        [0, 3, 7, 10, 14],  # Minor thirteenth

        # Added Tone Chords
        [0, 4, 7, 14],      # Add9 (Major triad + 9th)
        [0, 3, 7, 14],      # Minor add9
        [0, 4, 7, 2],       # Add2 (similar to sus2)
        [0, 4, 7, 5],       # Add4

        # Altered Chords
        [0, 4, 7, 10, 13],  # Dominant 7th with added 13th
        [0, 4, 7, 10, 15],  # Dominant 7th with added #11
    ]

    # Choose a random chord type
    intervals = random.choice(chord_types)

    # Add chord notes with the same duration as the root note
    new_notes = []
    for interval in intervals[1:]:  # Skip root as it already exists
        chord_pitch = root_note['pitch'] + interval
        # Ensure pitch is in MIDI range
        if MIN_PITCH <= chord_pitch <= MAX_PITCH:
            # Slightly reduce velocity of harmony notes
            chord_velocity = int(root_note['velocity'] * 0.8)
            # Align chord start to the nearest beat division
            chord_start = quantize_time(root_note['start'])
            chord_end = quantize_time(chord_start + (root_note['end'] - root_note['start']))
            chord_note = {
                'pitch': chord_pitch,
                'start': chord_start,
                'end': chord_end,
                'velocity': chord_velocity
            }
            new_notes.append(chord_note)

    note_sequence.extend(new_notes)
    note_sequence.sort(key=lambda x: x['start'])

    return note_sequence



def time_stretch(note_sequence: NoteSequence, factor: float) -> NoteSequence:
    """
    Stretch the timing of all notes by a given factor,
    ensuring alignment with the beat grid.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.
        factor (float): Stretch factor (e.g., 1.1 to lengthen durations).

    Returns:
        NoteSequence: The time-stretched note sequence.
    """
    stretched = copy.deepcopy(note_sequence)
    for note in stretched:
        note['start'] *= factor
        note['end'] *= factor
        # Quantize to maintain rhythmic alignment
        note['start'] = quantize_time(note['start'])
        note['end'] = quantize_time(note['end'])
    return stretched


def add_random_note(note_sequence: NoteSequence) -> NoteSequence:
    """
    Add a random note to the sequence, aligned to the beat grid.
    """
    new_pitch = random.randint(MIN_PITCH, MAX_PITCH)
    new_start = random.uniform(0, MAX_TIME)
    new_start = quantize_time(new_start)
    new_end = new_start + random.uniform(MIN_NOTE_DURATION, MAX_NOTE_DURATION)
    new_end = quantize_time(new_end)
    new_velocity = random.randint(VELOCITY_RANGE[0], VELOCITY_RANGE[1])
    
    new_note = {
        'pitch': new_pitch,
        'start': new_start,
        'end': new_end,
        'velocity': new_velocity
    }
    
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

def modify_random_note_improved(note_sequence: NoteSequence) -> NoteSequence:
    """
    Modify a random attribute of a random note in the sequence,
    ensuring alignment to the beat grid and maintaining musical coherence.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.

    Returns:
        NoteSequence: The mutated note sequence.
    """
    if not note_sequence:
        return note_sequence
    idx = random.randint(0, len(note_sequence) - 1)
    note = note_sequence[idx]
    modification = random.choice(['pitch', 'velocity', 'duration'])
    
    if modification == 'pitch':
        # Apply a scale-conscious pitch shift
        semitones = random.choice([-2, -1, 1, 2])  # Small pitch shifts
        note['pitch'] += semitones
        note['pitch'] = max(MIN_PITCH, min(MAX_PITCH, note['pitch']))
    elif modification == 'velocity':
        # Adjust velocity within range
        change = random.choice([-10, -5, 5, 10])
        note['velocity'] += change
        note['velocity'] = max(VELOCITY_RANGE[0], min(VELOCITY_RANGE[1], note['velocity']))
    elif modification == 'duration':
        # Adjust duration within one beat
        change = random.choice([-BEAT_DURATION, BEAT_DURATION])
        new_end = note['end'] + change
        # Ensure duration is at least one beat and aligned to grid
        if new_end > note['start'] + BEAT_DURATION:
            note['end'] = quantize_time(new_end)
        else:
            note['end'] = note['start'] + BEAT_DURATION

    # Ensure end is after start
    if note['end'] <= note['start']:
        note['end'] = note['start'] + BEAT_DURATION

    # Quantize to maintain rhythm
    note['start'] = quantize_time(note['start'])
    note['end'] = quantize_time(note['end'])

    return note_sequence


def mutate(note_sequence: NoteSequence) -> NoteSequence:
    """
    Enhanced mutation function with musically informed operations.
    Focuses on modifying existing notes and structural mutations.

    Parameters:
        note_sequence (NoteSequence): The current note sequence.

    Returns:
        NoteSequence: The mutated note sequence.
    """
    mutation_type = random.choices(
        ['pitch_shift', 'time_stretch', 'modify_note', 'add_chord', 'mutate_phrases', 'mutate_motifs'],
        weights=[0.3, 0.2, 0.3, 0.1, 0.05, 0.05],
        k=1
    )[0]
    
    if mutation_type == 'add_chord':
        return add_chord(note_sequence)
    elif mutation_type == 'pitch_shift':
        # Define root and scale type, possibly inferred from the current sequence
        root = random.choice([60, 62, 64, 65, 67, 69, 71])  # C, D, E, F, G, A, B in MIDI
        scale_type = random.choice(['major', 'minor'])
        semitones = random.choice([-12, -7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 8, 9, 12])  # Common transpositions
        return pitch_shift_scale_conscious(note_sequence, semitones, root, scale_type)
    elif mutation_type == 'time_stretch':
        factor = random.uniform(0.9, 1.1)  # Small time stretch factors
        return time_stretch(note_sequence, factor)
    elif mutation_type == 'modify_note':
        return modify_random_note_improved(note_sequence)
    elif mutation_type == 'mutate_phrases':
        return mutate_phrases(note_sequence)
    elif mutation_type == 'mutate_motifs':
        return mutate_motifs(note_sequence)
    else:
        return note_sequence



# --------------------------- Crossover functions --------------------------- #

# Adjust timing to maintain chronological order
def adjust_timing(offspring: NoteSequence) -> NoteSequence:
    if not offspring:
        return offspring
    sorted_offspring = sorted(offspring, key=lambda x: x['start'])
    previous_end = 0.0
    for note in sorted_offspring:
        if note['start'] < previous_end:
            note['start'] = previous_end
            note['end'] = note['start'] + (note['end'] - note['start'])
        previous_end = note['end']
    return sorted_offspring

def segment_based_crossover(parent1: NoteSequence, parent2: NoteSequence, segment_length: int = 8) -> Tuple[NoteSequence, NoteSequence]:
    """
    Perform segment-based crossover between two parent note sequences, ensuring musical coherence.

    Parameters:
        parent1 (NoteSequence): The first parent note sequence.
        parent2 (NoteSequence): The second parent note sequence.
        segment_length (int): Number of consecutive notes to swap.

    Returns:
        Tuple[NoteSequence, NoteSequence]: Two offspring note sequences.
    """
    if not parent1 or not parent2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    # Sort parents by start time to ensure chronological order
    sorted_parent1 = sorted(parent1, key=lambda x: x['start'])
    sorted_parent2 = sorted(parent2, key=lambda x: x['start'])
    
    def get_segments(parent: NoteSequence, seg_length: int) -> List[NoteSequence]:
        segments = []
        for i in range(0, len(parent) - seg_length + 1, seg_length):
            segments.append(parent[i:i + seg_length])
        return segments
    
    segments1 = get_segments(sorted_parent1, segment_length)
    segments2 = get_segments(sorted_parent2, segment_length)
    
    if not segments1 or not segments2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    # Select random segments to swap
    seg1 = random.choice(segments1)
    seg2 = random.choice(segments2)
    
    # Find the start indices
    def find_segment_start(parent: NoteSequence, segment: NoteSequence) -> int:
        for i in range(len(parent) - len(segment) + 1):
            if parent[i:i + len(segment)] == segment:
                return i
        return 0  # Default to start if not found
    
    start1 = find_segment_start(sorted_parent1, seg1)
    start2 = find_segment_start(sorted_parent2, seg2)
    
    # Create offspring by swapping segments
    offspring1 = sorted_parent1[:start1] + seg2 + sorted_parent1[start1 + segment_length:]
    offspring2 = sorted_parent2[:start2] + seg1 + sorted_parent2[start2 + segment_length:]
    
    offspring1 = adjust_timing(offspring1)
    offspring2 = adjust_timing(offspring2)
    
    # Validate offspring
    offspring1 = validate_note_sequence(offspring1)
    offspring2 = validate_note_sequence(offspring2)
    
    return offspring1, offspring2


def single_point_crossover(parent1: NoteSequence, parent2: NoteSequence, phrase_length: int = 8, num_crossover_points: int = 1) -> Tuple[NoteSequence, NoteSequence]:
    """
    Perform single-point crossover at phrase boundaries between two parent note sequences.

    Parameters:
        parent1 (NoteSequence): The first parent note sequence.
        parent2 (NoteSequence): The second parent note sequence.
        phrase_length (int): Number of notes per phrase.
        num_crossover_points (int): Number of crossover points to perform.

    Returns:
        Tuple[NoteSequence, NoteSequence]: Two offspring note sequences.
    """
    if not parent1 or not parent2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    # Sort parents by start time
    sorted_parent1 = sorted(parent1, key=lambda x: x['start'])
    sorted_parent2 = sorted(parent2, key=lambda x: x['start'])
    
    # Define phrases
    def get_phrases(parent: NoteSequence, phrase_len: int) -> List[NoteSequence]:
        return [parent[i:i + phrase_len] for i in range(0, len(parent), phrase_len)]
    
    phrases1 = get_phrases(sorted_parent1, phrase_length)
    phrases2 = get_phrases(sorted_parent2, phrase_length)
    
    if not phrases1 or not phrases2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    # Determine crossover points based on number of crossover points
    crossover_indices1 = sorted(random.sample(range(1, len(phrases1)), min(num_crossover_points, len(phrases1)-1)))
    crossover_indices2 = sorted(random.sample(range(1, len(phrases2)), min(num_crossover_points, len(phrases2)-1)))
    
    # Create offspring by alternating phrases between parents
    def create_offspring(phrases_a: List[NoteSequence], phrases_b: List[NoteSequence], crossover_points: List[int]) -> NoteSequence:
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
    
    # Flatten the list of phrases
    offspring1 = [note for phrase in offspring1_phrases for note in phrase]
    offspring2 = [note for phrase in offspring2_phrases for note in phrase]
    
    offspring1 = adjust_timing(offspring1)
    offspring2 = adjust_timing(offspring2)
    
    # Validate offspring
    offspring1 = validate_note_sequence(offspring1)
    offspring2 = validate_note_sequence(offspring2)
    
    return offspring1, offspring2

def uniform_crossover(parent1: NoteSequence, parent2: NoteSequence, swap_probability: float = 0.5) -> Tuple[NoteSequence, NoteSequence]:
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
    
    # Append remaining notes from longer parent
    if len(sorted_parent1) > min_length:
        offspring1.extend(copy.deepcopy(sorted_parent1[min_length:]))
    if len(sorted_parent2) > min_length:
        offspring2.extend(copy.deepcopy(sorted_parent2[min_length:]))
    
    offspring1 = adjust_timing(offspring1)
    offspring2 = adjust_timing(offspring2)
    
    # Validate offspring
    offspring1 = validate_note_sequence(offspring1)
    offspring2 = validate_note_sequence(offspring2)
    
    return offspring1, offspring2




def harmonize_offspring(note_sequence: NoteSequence, key: int = 0, scale: List[int] = None) -> NoteSequence:
    """
    Adjust the offspring's pitch and rhythm to ensure harmonic and rhythmic coherence.

    Parameters:
        note_sequence (NoteSequence): The offspring's note sequence.
        key (int): The root note of the scale (MIDI note number, e.g., 60 for Middle C).
        scale (List[int]): The scale degrees (intervals in semitones) from the root. Default is C major.

    Returns:
        NoteSequence: Harmonized note sequence.
    """
    if scale is None:
        scale = [0, 2, 4, 5, 7, 9, 11]  # C major scale intervals

    # Sort notes by start time
    sorted_notes = sorted(note_sequence, key=lambda x: x['start'])
    
    previous_pitch = None
    for i, note in enumerate(sorted_notes):
        # **Scale Adherence**
        # Adjust pitch to fit within the scale
        relative_pitch = note['pitch'] - key
        scale_degree = relative_pitch % 12
        if scale_degree not in scale:
            # Find the nearest scale degree
            nearest = min(scale, key=lambda sd: abs(sd - scale_degree))
            adjusted_pitch = key + (relative_pitch - scale_degree) + nearest
            note['pitch'] = max(0, min(127, adjusted_pitch))
        
        # **Melodic Contour Preservation**
        if previous_pitch is not None:
            pitch_diff = note['pitch'] - previous_pitch
            if abs(pitch_diff) > MAX_PITCH_JUMP:
                # Reduce the pitch jump to within the threshold
                pitch_diff = MAX_PITCH_JUMP if pitch_diff > 0 else -MAX_PITCH_JUMP
                note['pitch'] = previous_pitch + pitch_diff
                note['pitch'] = max(0, min(127, note['pitch']))
        
        previous_pitch = note['pitch']
    
    # **Rhythmic Consistency**
    # Align note starts to a grid based on a common time signature (e.g., 4/4)
    BEAT_DURATION = 0.5  # seconds per beat
    GRID_RESOLUTION = 0.05  # allowable deviation in seconds

    for note in sorted_notes:
        aligned_start = round(note['start'] / BEAT_DURATION) * BEAT_DURATION
        if abs(note['start'] - aligned_start) > GRID_RESOLUTION:
            note['start'] = aligned_start
            note['end'] = note['start'] + max(0.1, note['end'] - note['start'])  # Ensure positive duration
    
    # **Dynamic Range Control**
    # Normalize velocities to a preferred range
    PREFERRED_VELOCITY_MIN = 60
    PREFERRED_VELOCITY_MAX = 100

    for note in sorted_notes:
        if note['velocity'] < PREFERRED_VELOCITY_MIN:
            note['velocity'] = PREFERRED_VELOCITY_MIN
        elif note['velocity'] > PREFERRED_VELOCITY_MAX:
            note['velocity'] = PREFERRED_VELOCITY_MAX
    
    # **Ensure No Overlapping Notes of Same Pitch**
    active_pitches = {}
    for note in sorted_notes:
        pitch = note['pitch']
        if pitch in active_pitches:
            if note['start'] < active_pitches[pitch]:
                # Overlapping note detected; adjust the start time
                note['start'] = active_pitches[pitch] + 0.1  # Shift by 0.1 seconds
                note['end'] = note['start'] + max(0.1, note['end'] - note['start'])
        active_pitches[pitch] = note['end']
    
    # Final sort and return
    harmonized_sequence = sorted(sorted_notes, key=lambda x: x['start'])
    
    return harmonized_sequence




# --------------------------- Enhanced Fitness Function --------------------------- #

def fitness(note_sequence: list) -> float:
    """
    Enhanced fitness function with normalized scores to prevent infinite values.
    Includes penalties for out-of-range notes and rewards harmonious intervals.
    Adds rhythmic consistency and dynamic balance.

    Parameters:
        note_sequence (list of dict): The note sequence to evaluate.

    Returns:
        float: Fitness score.
    """
    if not note_sequence:
        return 0.0

    # Define the initial weights for each fitness component (sum to less than 1.0)
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
        # Sort notes by start time once
        sorted_notes = sorted(note_sequence, key=lambda x: x['start'])
        total_notes = len(sorted_notes)

        # 1. Unique Pitches Score (0-1)
        unique_pitches = len(set(note['pitch'] for note in sorted_notes))
        unique_pitches_score = min(1.0, unique_pitches / 50.0)  # Adjust normalization as needed

        # 2. Total Notes Score (0-1)
        total_notes_score = min(1.0, total_notes / 500.0)  # Adjust normalization as needed

        # 3. Smoothness Score (0-1)
        smooth_transitions = 0
        total_transitions = total_notes - 1
        if total_transitions > 0:
            for i in range(total_transitions):
                current_note = sorted_notes[i]
                next_note = sorted_notes[i + 1]

                pitch_diff = abs(current_note['pitch'] - next_note['pitch'])
                time_diff = max(0.0001, next_note['start'] - current_note['end'])

                # Normalize pitch_diff to octave (12 semitones)
                octave_diff = pitch_diff % 12
                if octave_diff > 6:
                    octave_diff = 12 - octave_diff  # Smallest interval within an octave

                # Define thresholds
                MAX_PITCH_DIFF = 6  # Half an octave
                MAX_TIME_DIFF = 0.5  # seconds

                if octave_diff <= MAX_PITCH_DIFF and time_diff < MAX_TIME_DIFF:
                    smooth_transitions += 1

            smoothness_score = smooth_transitions / total_transitions
        else:
            smoothness_score = 0.0

        # 4. Harmony Score (0-1)
        harmonious_intervals = 0
        acceptable_intervals = {0, 3, 4, 5, 7, 8, 9, 12}  # Unison, minor/major thirds, etc.
        total_intervals = total_transitions
        if total_intervals > 0:
            for i in range(total_transitions):
                current_pitch = sorted_notes[i]['pitch']
                next_pitch = sorted_notes[i + 1]['pitch']
                interval = abs(current_pitch - next_pitch) % 12  # Modulo octave

                if interval in acceptable_intervals:
                    harmonious_intervals += 1

            harmony_score = harmonious_intervals / total_intervals
        else:
            harmony_score = 0.0

        # 5. Phrase Structure Score (0-1)
        phrase_score = 0.0
        total_possible_phrases = max(1, total_notes // IDEAL_PHRASE_LENGTH)
        phrase_boundaries = 0

        for i in range(1, total_notes):
            time_gap = sorted_notes[i]['start'] - sorted_notes[i - 1]['end']
            # Define reasonable phrase boundary
            if 0.1 <= time_gap <= 0.5:
                phrase_boundaries += 1

        phrase_score = min(1.0, phrase_boundaries / total_possible_phrases)

        # 6. Rhythmic Consistency Score (0-1)
        aligned_notes = sum(
            1 for note in sorted_notes
            if abs((note['start'] / BEAT_DURATION) - round(note['start'] / BEAT_DURATION)) < 0.05
        )
        rhythmic_consistency_score = aligned_notes / total_notes if total_notes > 0 else 0.0

        # 7. Dynamic Balance Score (0-1)
        velocities = [note['velocity'] for note in sorted_notes]
        mean_velocity = np.mean(velocities) if velocities else 0.0
        dynamic_balance_score = 1 - abs(mean_velocity - PREFERRED_VELOCITY) / PREFERRED_VELOCITY
        dynamic_balance_score = max(0.0, min(1.0, dynamic_balance_score))

        # 8. Additional Penalties (e.g., out-of-range notes)
        out_of_range = sum(
            1 for note in sorted_notes if note['pitch'] < MIN_PITCH or note['pitch'] > MAX_PITCH
        )
        if total_notes > 0:
            out_of_range_penalty = 1 - (out_of_range / total_notes)  # Penalize if any notes out of range
        else:
            out_of_range_penalty = 1.0

        # 9. Legato Score (0-1)
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

        # 10. Key Consistency Score (0-1)
        key, scale_type, key_consistency_score = determine_key(sorted_notes)

        # Update weights to include new components
        additional_weights = {
            'legato': 0.10,
            'key_consistency': 0.10
        }
        WEIGHTS.update(additional_weights)

        # Normalize weights to sum to 1.0
        total_weight = sum(WEIGHTS.values())
        WEIGHTS = {k: v / total_weight for k, v in WEIGHTS.items()}

        # Calculate the final score
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

        # Clamp final score to [0, MAX_SCORE]
        final_score = max(0.0, min(MAX_SCORE, final_score))

        return final_score

    except Exception as e:
        print(f"Error in fitness calculation: {e}")
        return 0.0




# --------------------------- Evolutionary Algorithm --------------------------- #

def initialize_population(midi_dataset: List[pretty_midi.PrettyMIDI], population_size: int, max_initial_files: int = 500) -> List[NoteSequence]:
    """
    Initialize the population by selecting random note sequences from the MIDI dataset.
    Only piano notes are considered.
    """
    population = []
    limited_dataset = midi_dataset[:max_initial_files]
    
    for _ in range(population_size):
        midi = random.choice(limited_dataset)
        combined_notes = []
        
        # Combine notes only from piano instruments
        for instrument in midi.instruments:
            # Check if the instrument is a piano
            if instrument.program == 0 or 'piano' in instrument.name.lower():
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

def create_next_generation(selected: List[NoteSequence],
                           population_size: int,
                           mutation_rate: float,
                           crossover_rate: float = 0.8,
                           crossover_types: Dict[str, float] = {'single_point': 0.33, 'segment_based': 0.33, 'uniform': 0.34}) -> List[NoteSequence]:
    """
    Create the next generation using crossover and mutation.
    Includes limiting note density to prevent overcrowding.

    Parameters:
        selected (List[NoteSequence]): Selected top-performing individuals.
        population_size (int): Desired population size.
        mutation_rate (float): Probability of mutation for each offspring.
        crossover_rate (float): Probability of applying crossover.
        crossover_type (str): Type of crossover ('single_point' or 'segment_based').

    Returns:
        List[NoteSequence]: The next generation of note sequences.
    """
    next_generation = []
    max_attempts = population_size * 2
    attempts = 0

    while len(next_generation) < population_size and attempts < max_attempts:
        parent1, parent2 = random.sample(selected, 2)

        # Apply Crossover
        if random.random() < crossover_rate:
            if crossover_types== 'single_point':
                offspring1, offspring2 = single_point_crossover(parent1, parent2)
            elif crossover_types == 'segment_based':
                offspring1, offspring2 = segment_based_crossover(parent1, parent2)
            elif crossover_types == 'uniform':
                offspring1, offspring2 = uniform_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        else:
            offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Harmonize offspring (if needed)
        offspring1 = harmonize_offspring(offspring1)
        offspring2 = harmonize_offspring(offspring2)

        # Apply Mutation
        if random.random() < mutation_rate:
            offspring1 = mutate(offspring1)
        if random.random() < mutation_rate:
            offspring2 = mutate(offspring2)

        # Validate and Quantize
        offspring1 = validate_note_sequence(offspring1)
        offspring2 = validate_note_sequence(offspring2)

        # Apply Velocity Smoothing and Phrase Dynamics
        offspring1 = smooth_velocities(offspring1)
        offspring1 = apply_phrase_dynamics(offspring1)
        offspring1 = apply_accentuation(offspring1)
        offspring2 = smooth_velocities(offspring2)
        offspring2 = apply_phrase_dynamics(offspring2)
        offspring2 = apply_accentuation(offspring2)

        # Apply Voice Leading
        offspring1 = apply_voice_leading(offspring1)
        offspring2 = apply_voice_leading(offspring2)

        # Apply Rhythm Enhancements (Swing Feel)
        offspring1 = apply_swing(offspring1)
        offspring2 = apply_swing(offspring2)

        # Apply Dynamic Balance and Accentuation
        offspring1 = apply_dynamic_balance(offspring1)
        offspring2 = apply_dynamic_balance(offspring2)

        # Add offspring to next generation
        if offspring1:
            next_generation.append(offspring1)
        if len(next_generation) < population_size and offspring2:
            next_generation.append(offspring2)

        attempts += 1

    # If not enough offspring, fill the rest with mutated copies of selected individuals
    while len(next_generation) < population_size and selected:
        offspring = copy.deepcopy(random.choice(selected))
        if random.random() < mutation_rate:
            offspring = mutate(offspring)
        # Apply post-mutation transformations
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
    Run the evolutionary algorithm with crossover and mutation.

    Parameters:
        midi_dataset (List[pretty_midi.PrettyMIDI]): Loaded MIDI dataset.
        generations (int): Number of generations to evolve.
        population_size (int): Number of individuals in the population.
        selection_size (int): Number of top individuals to select.
        mutation_rate (float): Probability of mutation.
        crossover_rate (float): Probability of applying crossover.
        crossover_type (str): Type of crossover ('single_point' or 'segment_based').

    Returns:
        NoteSequence: The best evolved note sequence.
    """
    try:
        population = initialize_population(midi_dataset, population_size)
        if not population:
            logging.error("Failed to initialize population")
            return []
        
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(1, generations + 1):
            try:
                # Evaluate fitness
                fitnesses = []
                for individual in population:
                    try:
                        fit = fitness(individual)
                        if fit == float('inf') or np.isnan(fit):
                            fit = 0.0
                        fitnesses.append(fit)
                    except Exception as e:
                        logging.error(f"Error calculating fitness: {e}")
                        fitnesses.append(0.0)
                
                current_best = max(fitnesses) if fitnesses else 0.0
                current_avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
                
                logging.info(f"Generation {generation}: Best Fitness = {current_best:.2f}, Avg Fitness = {current_avg:.2f}")
                
                # Update best individual
                if current_best > best_fitness and fitnesses:
                    best_fitness = current_best
                    best_individual = population[fitnesses.index(current_best)]
                
                # Selection
                selected = select_best(population, fitnesses, selection_size)
                if not selected:
                    logging.warning("Selection failed, restarting population")
                    population = initialize_population(midi_dataset, population_size)
                    continue
                
                # Create next generation with crossover and mutation
                population = create_next_generation(
                    selected=selected,
                    population_size=population_size,
                    mutation_rate=mutation_rate,
                    crossover_rate=crossover_rate,
                    crossover_types=crossover_types
                )
                if not population:
                    logging.warning("Failed to create next generation, restarting population")
                    population = initialize_population(midi_dataset, population_size)
                
            except Exception as e:
                logging.error(f"Error in generation {generation}: {e}")
                population = initialize_population(midi_dataset, population_size)
        
        return best_individual if best_individual else population[0] if population else []
    
    except Exception as e:
        logging.fatal(f"Fatal error in evolutionary music generation: {e}")
        return []


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
    print("Crafting superb music.")
    best_music = evolutionary_music_generation(
        midi_dataset=midi_dataset,
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        selection_size=SELECTION_SIZE,
        mutation_rate=MUTATION_RATE
    )

    # Step 3: Convert the best individual back to MIDI and save
    print("Converting the best individual to MIDI...")
    #best_music = eliminate_long_silences(best_music, max_silence_duration=1.0)
    #best_music = remove_long_single_notes(best_music, max_note_duration=1.0)
    output_midi = note_sequence_to_midi(best_music)
    output_midi.write(OUTPUT_MIDI_PATH)
    print(f"Evolved music saved as '{OUTPUT_MIDI_PATH}'.")

if __name__ == "__main__":
    main()
