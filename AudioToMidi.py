import soundfile
import resampy
import os
import vamp
import numpy as np
from midiutil.MidiFile import MIDIFile
from scipy.signal import medfilt
import jams

# Replace 'path/to/conda/envs/your_environment_name' with the path to your Conda environment
"""
Many thanks to https://github.com/justinsalamon/audio_to_midi_melodia for the original script.
"""

class AudioToMidiMelodia:

    def __init__(self, smooth=0.25, minduration=0.1):
        self.smooth = smooth
        self.minduration = minduration

    def save_jams(self, jamsfile, notes, track_duration, orig_filename):

        # Construct a new JAMS object and annotation records
        jam = jams.JAMS()

        # Store the track duration
        jam.file_metadata.duration = track_duration
        jam.file_metadata.title = orig_filename

        midi_an = jams.Annotation(namespace='pitch_midi',
                                duration=track_duration)
        midi_an.annotation_metadata = \
            jams.AnnotationMetadata(
                data_source='audio_to_midi_melodia.py v%s' % __init__.__version__,
                annotation_tools='audio_to_midi_melodia.py (https://github.com/'
                                'justinsalamon/audio_to_midi_melodia)')

        # Add midi notes to the annotation record.
        for n in notes:
            midi_an.append(time=n[0], duration=n[1], value=n[2], confidence=0)

        # Store the new annotation in the jam
        jam.annotations.append(midi_an)

        # Save to disk
        jam.save(jamsfile)

    def save_midi(self, outfile, notes, tempo):

        track = 0
        time = 0
        midifile = MIDIFile(1)

        # Add track name and tempo.
        midifile.addTrackName(track, time, "MIDI TRACK")
        midifile.addTempo(track, time, tempo)

        channel = 0
        volume = 100

        for note in notes:
            onset = note[0] * (tempo/60.)
            duration = note[1] * (tempo/60.)
            # duration = 1
            pitch = note[2]
            pitch = pitch.__int__()
            midifile.addNote(track, channel, pitch, onset, duration, volume)

        # And write it to disk.
        binfile = open(outfile, 'wb')
        midifile.writeFile(binfile)
        binfile.close()

    def midi_to_notes(self, midi, fs, hop):

        # smooth midi pitch sequence first
        if (self.smooth > 0):
            filter_duration = self.smooth  # in seconds
            filter_size = int(filter_duration * fs / float(hop))
            if filter_size % 2 == 0:
                filter_size += 1
            midi_filt = medfilt(midi, filter_size)
        else:
            midi_filt = midi
        # print(len(midi),len(midi_filt))

        notes = []
        p_prev = 0
        duration = 0
        onset = 0
        for n, p in enumerate(midi_filt):
            if p == p_prev:
                duration += 1
            else:
                # treat 0 as silence
                if p_prev > 0:
                    # add note
                    duration_sec = duration * hop / float(fs)
                    # only add notes that are long enough
                    if duration_sec >= self.minduration:
                        onset_sec = onset * hop / float(fs)
                        notes.append((onset_sec, duration_sec, p_prev))

                # start new note
                onset = n
                duration = 1
                p_prev = p

        # add last note
        if p_prev > 0:
            # add note
            duration_sec = duration * hop / float(fs)
            onset_sec = onset * hop / float(fs)
            notes.append((onset_sec, duration_sec, p_prev))

        return notes

    def hz2midi(self, hz):

        # convert from Hz to midi note
        hz_nonneg = hz.copy()
        idx = hz_nonneg <= 0
        hz_nonneg[idx] = 1
        midi = 69 + 12*np.log2(hz_nonneg/440.)
        midi[idx] = 0

        # round
        midi = np.round(midi)

        return midi

    def format_midi(self, notes):
        out = []
        for n in notes:
            out.append({'time':str(round(n[0],2)), 'duration':str(round(n[1],2)), 'value':str(round(n[2],2))})
        return out

    def audio_to_midi_melodia(self, infile, bpm, sr=None, outfile=None, savejams=False, return_format=0):

        # define analysis parameters
        fs = 44100
        hop = 128

        os.environ['VAMP_PATH'] = '/home/caleb/miniconda3/envs/stl/vamp_plugins'

        print("Loading audio...")
        if isinstance(infile, str):
            data, sr = soundfile.read(infile)
        elif sr is not None:
            data = infile
            sr = sr
        else:
            raise ValueError("Invalid input: either provide a file path or y and sr values.")

        # mixdown to mono if needed
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data.mean(axis=1)
        # resample to 44100 if needed
        if sr != fs:
            data = resampy.resample(data, sr, fs)
            sr = fs

        # extract melody using melodia vamp plugin
        print("Extracting melody f0 with MELODIA...")
        melody = vamp.collect(data, sr, "mtg-melodia:melodia",
                            parameters={"voicing": 0.2})

        # hop = melody['vector'][0]
        pitch = melody['vector'][1]

        # impute missing 0's to compensate for starting timestamp
        pitch = np.insert(pitch, 0, [0]*8)

        # debug
        # np.asarray(pitch).dump('f0.npy')
        # print(len(pitch))

        # convert f0 to midi notes
        print("Converting Hz to MIDI notes...")
        midi_pitch = self.hz2midi(pitch)

        # segment sequence into individual midi notes
        notes = self.midi_to_notes(midi_pitch, fs, hop)

        if savejams:
            print("Saving JAMS to disk...")
            jamsfile = os.path.splitext(outfile)[0] + ".jams"
            track_duration = len(data) / float(fs)
            self.save_jams(jamsfile, notes, track_duration, os.path.basename(infile))

        if return_format == 0:
            return self.format_midi(notes)
        elif return_format == 1:
            print("Saving MIDI to disk...")
            self.save_midi(outfile, notes, bpm)
            return self.format_midi(notes)
        elif return_format == 2:
            print("Saving MIDI to disk...")
            self.save_midi(outfile, notes, bpm)
            return

