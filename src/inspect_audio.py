import os
import librosa

AUDIO_DIR = "data/audio/audio_speech_actors_01-24"

emotion_names = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

counts = {emotion: 0 for emotion in emotion_names.values()}
durations = []
errors = []

for actor in sorted(os.listdir(AUDIO_DIR)):
    actor_path = os.path.join(AUDIO_DIR, actor)

    if not os.path.isdir(actor_path):
        continue

    for filename in sorted(os.listdir(actor_path)):
        if not filename.endswith(".wav"):
            continue

        parts = filename.replace(".wav", "").split("-")

        if len(parts) != 7:
            errors.append(filename)
            continue

        emotion_code = parts[2]
        emotion = emotion_names.get(emotion_code)

        if emotion is None:
            errors.append(filename)
            continue

        path = os.path.join(actor_path, filename)

        try:
            duration = librosa.get_duration(path=path)
            durations.append(duration)
            counts[emotion] += 1
        except Exception:
            errors.append(filename)

print("=" * 60)
print("AUDIO DATASET INSPECTION")
print("=" * 60)

print("\nEmotion counts:")
for emotion, count in counts.items():
    print(f"{emotion}: {count}")

print(f"\nTotal valid files: {sum(counts.values())}")
print(f"Total duration: {sum(durations) / 60:.2f} minutes")
print(f"Average duration: {sum(durations) / len(durations):.2f} seconds")
print(f"Files with errors: {len(errors)}")

if errors:
    print("\nFirst 10 errors:")
    for filename in errors[:10]:
        print(filename)