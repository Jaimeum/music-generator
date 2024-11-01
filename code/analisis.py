import os
from mido import MidiFile
from music21 import converter, note, chord, meter, tempo, key, analysis
import matplotlib.pyplot as plt
from pygame import mixer
import time
from collections import Counter
import numpy as np

class MidiAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        mixer.init()
    
    def find_band(self, band_name):
        """Search for a band by name (case-insensitive)"""
        all_bands = [d for d in os.listdir(self.dataset_path) 
                    if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        matches = [band for band in all_bands 
                  if band_name.lower() in band.lower()]
        return matches

    def list_songs(self, band_name):
        """List all MIDI files for a specific band"""
        band_path = os.path.join(self.dataset_path, band_name)
        return [f for f in os.listdir(band_path) 
                if f.lower().endswith(('.mid', '.midi'))]

    def play_midi(self, band_name, song_name):
        """Play the selected MIDI file"""
        midi_path = os.path.join(self.dataset_path, band_name, song_name)
        try:
            mixer.music.load(midi_path)
            mixer.music.play()
            print("\nPlaying MIDI file... Press Ctrl+C to stop.")
            
            # Keep the music playing until interrupted
            try:
                while mixer.music.get_busy():
                    time.sleep(1)
            except KeyboardInterrupt:
                mixer.music.stop()
                print("\nPlayback stopped.")
        except Exception as e:
            print(f"Error playing MIDI file: {e}")
        finally:
            mixer.music.unload()

    def analyze_midi(self, band_name, song_name):
            """Analyze and visualize the MIDI file structure"""
            midi_path = os.path.join(self.dataset_path, band_name, song_name)
            
            # Load MIDI file
            print("\n=== MIDI Structure Analysis ===")
            mid = MidiFile(midi_path)
            print(f"Format: Type {mid.type}")
            print(f"Number of tracks: {len(mid.tracks)}")
            print(f"Ticks per beat: {mid.ticks_per_beat}")
            
            # Load with music21
            score = converter.parse(midi_path)
            
            # Basic Musical Analysis
            self._analyze_basic_properties(score)
            
            # Rhythmic Analysis
            self._analyze_rhythm(score)
            
            # Melodic Analysis
            notes = self._analyze_melody(score)
            
            # Harmonic Analysis
            self._analyze_harmony(score)
            
            # Track Analysis
            self._analyze_tracks(mid)
            
            # Visualizations
            self._create_visualizations(score, notes, song_name)
            
            # Show all plots and wait for user input
            plt.show(block=False)
            input("\nPress Enter to continue...")
            plt.close('all')
            
    def _analyze_basic_properties(self, score):
        """Analyze basic musical properties"""
        print("\n=== Basic Musical Properties ===")
        
        # Key signature
        key_sig = score.analyze('key')
        print(f"Key: {key_sig.tonic.name} {key_sig.mode}")
        
        # Time signature
        time_sigs = list(score.recurse().getElementsByClass(meter.TimeSignature))
        if time_sigs:
            print(f"Time Signature: {time_sigs[0]}")
        
        # Tempo
        tempos = list(score.recurse().getElementsByClass(tempo.MetronomeMark))
        if tempos:
            print(f"Tempo: {tempos[0].number} BPM")
        
        # Duration
        print(f"Duration: {score.duration.quarterLength:.2f} quarter notes")

    def _analyze_rhythm(self, score):
        """Analyze rhythmic patterns"""
        print("\n=== Rhythmic Analysis ===")
        
        # Get all note durations
        durations = [n.duration.quarterLength for n in score.recurse().notes]
        
        if durations:
            # Calculate rhythm statistics
            print(f"Most common note duration: {Counter(durations).most_common(1)[0][0]} quarter notes")
            print(f"Average note duration: {np.mean(durations):.2f} quarter notes")
            print(f"Total number of notes: {len(durations)}")
            
            # Identify rhythmic patterns
            rhythm_patterns = []
            for measure in score.recurse().getElementsByClass('Measure'):
                pattern = []
                for n in measure.notes:
                    pattern.append(n.duration.quarterLength)
                if pattern:
                    rhythm_patterns.append(tuple(pattern))
            
            # Show most common patterns
            common_patterns = Counter(rhythm_patterns).most_common(3)
            print("\nMost common rhythmic patterns (in quarter notes):")
            for pattern, count in common_patterns:
                print(f"Pattern {pattern}: appears {count} times")

    def _analyze_melody(self, score):
        """Analyze melodic elements"""
        print("\n=== Melodic Analysis ===")
        
        notes = []
        intervals = []
        prev_note = None
        
        for element in score.recurse().notes:
            if isinstance(element, note.Note):
                notes.append(element.nameWithOctave)
                if prev_note:
                    interval = element.pitch.midi - prev_note.pitch.midi
                    intervals.append(abs(interval))
                prev_note = element
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element))
        
        if intervals:
            print(f"Average interval size: {np.mean(intervals):.2f} semitones")
            print(f"Largest interval: {max(intervals)} semitones")
            print(f"Most common interval: {Counter(intervals).most_common(1)[0][0]} semitones")
        
        # Pitch range
        midi_notes = [n.pitch.midi for n in score.recurse().getElementsByClass(note.Note)]
        if midi_notes:
            print(f"Pitch range: {max(midi_notes) - min(midi_notes)} semitones")
            print(f"Lowest note: {note.Note(midi=min(midi_notes)).nameWithOctave}")
            print(f"Highest note: {note.Note(midi=max(midi_notes)).nameWithOctave}")
        
        return notes

    def _analyze_harmony(self, score):
        """Analyze harmonic content"""
        print("\n=== Harmonic Analysis ===")
        
        # Analyze chords
        chords = list(score.recurse().getElementsByClass(chord.Chord))
        if chords:
            # Count chord types
            chord_types = Counter(c.commonName for c in chords)
            print("\nMost common chord types:")
            for chord_type, count in chord_types.most_common(5):
                print(f"{chord_type}: {count} times")
        
        # Analyze key changes using a simpler windowing approach
        notes = list(score.recurse().notes)
        window_size = 20
        step_size = 10
        key_changes = []
        
        # Only proceed if we have enough notes
        if len(notes) >= window_size:
            for i in range(0, len(notes) - window_size + 1, step_size):
                window = notes[i:i + window_size]
                # Create a temporary stream with the window of notes
                window_stream = score.cloneEmpty()
                for n in window:
                    window_stream.append(n)
                # Analyze the key of this window
                key_analysis = window_stream.analyze('key')
                if key_analysis:
                    key_changes.append(key_analysis)
        
        if key_changes:
            print("\nKey progression:")
            current_key = None
            for k in key_changes:
                if k != current_key:
                    current_key = k
                    print(f"-> {k}")

    def _analyze_tracks(self, mid):
        """Analyze MIDI tracks"""
        print("\n=== Track Analysis ===")
        
        for i, track in enumerate(mid.tracks):
            # Count messages by type
            msg_types = Counter(msg.type for msg in track)
            
            print(f"\nTrack {i}: {track.name if hasattr(track, 'name') else 'Unnamed'}")
            print("Message types:")
            for msg_type, count in msg_types.most_common():
                print(f"  {msg_type}: {count}")

    def _create_visualizations(self, score, notes, song_name):
        """Create all visualizations"""
        # Create figure with subplots
        plt.figure(figsize=(20, 10))
        
        # Note distribution
        plt.subplot(2, 2, 1)
        note_counts = Counter(notes).most_common(20)
        notes_list, counts = zip(*note_counts)
        plt.bar(range(len(counts)), counts, color='skyblue')
        plt.xticks(range(len(counts)), notes_list, rotation=45)
        plt.title('Most Common Notes/Chords')
        plt.xlabel('Note/Chord')
        plt.ylabel('Frequency')
        
        # Note timeline
        plt.subplot(2, 2, 2)
        times = []
        pitches = []
        for n in score.recurse().notes:
            if isinstance(n, note.Note):
                times.append(float(n.offset))
                pitches.append(n.pitch.midi)
            elif isinstance(n, chord.Chord):
                times.extend([float(n.offset)] * len(n))
                pitches.extend([p.midi for p in n.pitches])
        plt.scatter(times, pitches, alpha=0.2, s=10)
        plt.title('Note Timeline')
        plt.xlabel('Time (quarters)')
        plt.ylabel('MIDI Pitch')
        
        # Rhythm distribution
        plt.subplot(2, 2, 3)
        durations = [n.duration.quarterLength for n in score.recurse().notes]
        if durations:
            plt.hist(durations, bins=20, color='lightgreen')
            plt.title('Note Duration Distribution')
            plt.xlabel('Duration (quarter notes)')
            plt.ylabel('Count')
        
        # Interval distribution
        plt.subplot(2, 2, 4)
        intervals = []
        prev_note = None
        for n in score.recurse().getElementsByClass(note.Note):
            if prev_note:
                interval = n.pitch.midi - prev_note.pitch.midi
                intervals.append(abs(interval))
            prev_note = n
        if intervals:
            plt.hist(intervals, bins=range(max(intervals)+2), color='salmon')
            plt.title('Interval Distribution')
            plt.xlabel('Interval Size (semitones)')
            plt.ylabel('Count')
        
        plt.tight_layout()


    def _plot_note_distribution(self, notes, song_name):
        """Create a bar plot of most common notes"""
        note_counts = Counter(notes).most_common(20)
        notes_list, counts = zip(*note_counts)
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(counts)), counts, color='skyblue')
        plt.xticks(range(len(counts)), notes_list, rotation=45)
        plt.title('Most Common Notes/Chords')
        plt.xlabel('Note/Chord')
        plt.ylabel('Frequency')
        plt.tight_layout()

    def _plot_note_timeline(self, score, song_name):
        """Create a scatter plot of notes over time"""
        plt.subplot(1, 2, 2)
        
        times = []
        pitches = []
        
        # Use recurse() instead of flat
        for n in score.recurse().notes:
            if isinstance(n, note.Note):
                times.append(float(n.offset))
                pitches.append(n.pitch.midi)
            elif isinstance(n, chord.Chord):
                times.extend([float(n.offset)] * len(n))
                pitches.extend([p.midi for p in n.pitches])
        
        plt.scatter(times, pitches, alpha=0.2, s=10)
        plt.title('Note Timeline')
        plt.xlabel('Time (quarters)')
        plt.ylabel('MIDI Pitch')
        plt.tight_layout()
        
    

def main():
    print("=== MIDI File Analyzer ===")
    
    # Get dataset path
    dataset_path = '/Users/jaimeuria/.cache/kagglehub/datasets/imsparsh/lakh-midi-clean/versions/1'
    
    analyzer = MidiAnalyzer(dataset_path)
    
    while True:
        # Search for band
        band_search = input("\nEnter band name to search (or 'quit' to exit): ").strip()
        if band_search.lower() == 'quit':
            break
        
        matches = analyzer.find_band(band_search)
        if not matches:
            print("No matching bands found.")
            continue
        
        # Show matching bands
        print("\nMatching bands:")
        for i, band in enumerate(matches, 1):
            print(f"{i}. {band}")
        
        # Select band
        try:
            band_idx = int(input("Select band number: ")) - 1
            if band_idx < 0 or band_idx >= len(matches):
                print("Invalid selection")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        selected_band = matches[band_idx]
        
        # Show songs
        songs = analyzer.list_songs(selected_band)
        if not songs:
            print("No MIDI files found for this band.")
            continue
        
        print("\nAvailable songs:")
        for i, song in enumerate(songs, 1):
            print(f"{i}. {song}")
        
        # Select song
        try:
            song_idx = int(input("Select song number: ")) - 1
            if song_idx < 0 or song_idx >= len(songs):
                print("Invalid selection")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        selected_song = songs[song_idx]
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Analyze MIDI file")
            print("2. Play MIDI file")
            print("3. Choose another song")
            print("4. Choose another band")
            
            try:
                choice = int(input("Enter your choice (1-4): "))
                if choice == 1:
                    print("\nAnalyzing MIDI file...")
                    analyzer.analyze_midi(selected_band, selected_song)
                elif choice == 2:
                    analyzer.play_midi(selected_band, selected_song)
                elif choice == 3:
                    break
                elif choice == 4:
                    break
                else:
                    print("Invalid choice")
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

if __name__ == "__main__":
    main()