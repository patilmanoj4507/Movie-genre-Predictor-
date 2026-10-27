
import joblib

model = joblib.load("movie_genre_model.pkl")

print("\n=== MOVIE GENRE PREDICTION SYSTEM ===")
print("Type 'exit' to close the program.\n")

while True:
    plot = input("Enter movie plot: ").strip()

    if plot.lower() == "exit":
        print("Program closed.")
        break

    if not plot:
        print("Please enter a movie plot.")
        continue

    genre = model.predict([plot])[0]
    print("Predicted Genre:", genre)
    print("-" * 40)
