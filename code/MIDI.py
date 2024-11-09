# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:39:22 2024

@author: ivanv
"""

import os
from mido import MidiFile
from music21 import converter, note, chord, meter, tempo, key, analysis
import matplotlib.pyplot as plt
from pygame import mixer
import time
from collections import Counter
import numpy as np
import random
from copy import deepcopy

class MidiAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        mixer.init()
        self.historial_caracteristicas = []  # Guarda las características de generaciones anteriores para penalizar repeticiones
    
    def calcular_fitness(self, score):
        """Calcula un puntaje de aptitud para un archivo MIDI con penalización de repetición."""
        fitness = 0
        
        # 1. Complejidad armónica (número de acordes únicos)
        chords = list(score.chordify().recurse().getElementsByClass('Chord'))
        unique_chords = len(set(str(c) for c in chords))
        fitness += unique_chords
        
        # 2. Rango melódico
        midi_notes = [n.pitch.midi for n in score.recurse().getElementsByClass(note.Note)]
        if midi_notes:
            range_melodico = max(midi_notes) - min(midi_notes)
            fitness += range_melodico / 2  # Escala el rango para mantener el puntaje balanceado
        
        # 3. Variedad rítmica (cuántas duraciones únicas se encuentran)
        durations = set(n.duration.quarterLength for n in score.recurse().notes)
        fitness += len(durations)
        
        # 4. Factor de creatividad (aleatoriedad en la calificación)
        creatividad = random.uniform(0, 2)  # Añade un pequeño factor aleatorio para fomentar la creatividad
        fitness += creatividad
        
        # Guardar características principales del individuo actual
        caracteristicas_individuo = (unique_chords, range_melodico, len(durations))
        
        # Penalizar similitud con generaciones anteriores
        if caracteristicas_individuo in self.historial_caracteristicas:
            fitness *= 0.8  # Aplica una penalización si es similar a algún individuo anterior

        # Agregar al historial para futuras comparaciones
        self.historial_caracteristicas.append(caracteristicas_individuo)
        
        return fitness

    def mutar_midi(self, midi_file):
        """Mutación más diversa en el archivo MIDI."""
        nuevo_midi = deepcopy(midi_file)
        for track in nuevo_midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and random.random() < 0.1:
                    msg.note += random.choice([-1, 1, -2, 2])  # Mutación de semitonos más variada
                if msg.type == 'set_tempo' and random.random() < 0.05:
                    msg.tempo = int(msg.tempo * random.uniform(0.85, 1.15))  # Tempo más variable
        return nuevo_midi
    
    def algoritmo_evolutivo(self, generacion_maxima=5, tamano_poblacion=10, seleccion=5):
        """Ejecuta el algoritmo evolutivo en la base de datos de canciones MIDI."""
        
        # Generar la población inicial
        midi_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.mid') or f.endswith('.midi')]
        poblacion = [self.mutar_midi(MidiFile(os.path.join(self.dataset_path, mf))) for mf in midi_files[:tamano_poblacion]]
        
        for generacion in range(generacion_maxima):
            fitness_scores = []
            
            # Calcular fitness de cada individuo en la población
            for midi in poblacion:
                score = converter.parse(midi)  # Cargar con music21
                fitness = self.calcular_fitness(score)
                fitness_scores.append((midi, fitness))
            
            # Seleccionar los mejores para reproducirse
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            padres = [x[0] for x in fitness_scores[:seleccion]]
            
            # Generar nueva población a partir de los padres seleccionados
            nueva_poblacion = []
            while len(nueva_poblacion) < tamano_poblacion:
                padre = random.choice(padres)
                hijo = self.mutar_midi(padre)  # Aplicar mutación
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion  # Actualizar la población
            print(f"Generación {generacion + 1} completada.")
        
        # Guardar la población final
        for i, midi_file in enumerate(poblacion):
            midi_file.save(os.path.join(self.dataset_path, f'cancion_evolucionada_{i}.mid'))
            print(f'Archivo guardado: cancion_evolucionada_{i}.mid')

# Ejecución del código
dataset_path = '/ruta'
analyzer = MidiAnalyzer(dataset_path)

# Ejecutar el algoritmo evolutivo
analyzer.algoritmo_evolutivo(generacion_maxima=5, tamano_poblacion=10, seleccion=5)
