### random qustions
1. "What does MLP stand for in MLPClassifier?"

    The Answer: It stands for Multi-Layer Perceptron.

    Why they ask: It’s the literal name of the model you are using. If you don't know what the acronym means, it's an immediate red flag.

2. "Why did you use random_state=42? Is 42 a magic number?"

    The Answer: No, 42 isn't magic. random_state sets the seed for the random number generator so the network initializes its weights the exact same way every time we run the script, making the results reproducible. You could use 0, 1, or 123. We just use 42 because it’s a long-running programmer joke from The Hitchhiker's Guide to the Galaxy.

    Why they ask: People who copy code from tutorials always leave 42 in there without knowing why. Explaining the joke proves you understand the parameter.

3. "Why are the variables named X_train and y_train? Why is the X capitalized and the y lowercase?"

    The Answer: X_train holds the inputs (the symptoms), and y_train holds the targets/labels (the diagnosis). By standard mathematical and data science convention, X is capitalized because it represents a 2D matrix (multiple rows and columns), while y is lowercase because it's a 1D vector (just a single list of answers).

    Why they ask: This is a core convention in Python machine learning. Knowing this shows you understand data structures.

4. "Why did you use [0, 1, 1, 0] instead of just writing out the words 'Fever' and 'Cough' in the array?"

    The Answer: Neural networks cannot read or process text; they only perform mathematical operations. We have to use binary encoding (or one-hot encoding) where 1 means the symptom is present and 0 means it's absent.

    Why they ask: To ensure you understand how data must be pre-processed before being fed to a machine learning model.

5. "In the line hidden_layer_sizes=(8,), what is the point of that comma inside the parentheses?"

    The Answer: It is specific Python syntax. We need to pass a "tuple" to that parameter. If we just write (8), Python interprets it as the integer 8 wrapped in parentheses. Writing (8,) forces Python to recognize it as a tuple containing one item.

    Why they ask: This is a strict Python language test. It proves you understand Python data types and didn't just blindly copy a weird-looking syntax.

6. "Where are the IF-THEN rules for the expert system in your code? I don't see them."

    The Answer: There aren't any! That is the entire point of using a neural network. Instead of hardcoding IF (Fever == 1) THEN (Diagnosis = Flu), the fit() function allows the neural network to automatically learn those logical connections by studying the X_train and y_train data.

    Why they ask: To see if you actually grasp the fundamental difference between traditional expert systems and machine learning.

7. "What exactly does the fit() function do in your script?"

    The Answer: That is the exact moment the training happens. The fit() function passes the symptoms (X) and the correct diagnoses (y) through the neural network, allowing it to adjust its internal weights to map the inputs to the outputs accurately.

    Why they ask: To make sure you know which part of the code is actually doing the "learning."

8. "How does the code calculate the 'Confidence Level' percentage?"

    The Answer: Instead of using the standard predict() method which just spits out one final answer, we also use predict_proba(). This outputs a probability array for all three possible diseases (e.g., [0.10, 0.85, 0.05]). We take the highest probability, multiply it by 100, and format it as a percentage.

    Why they ask: To verify you understand the difference between discrete classification and probabilistic outputs.
