from count_vectorizer import CountVectorizer

def test_feature_names_and_counts():
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)

    expected_features = [
        'crock', 'pot', 'pasta', 'never', 'boil', 'again',
        'pomodoro', 'fresh', 'ingredients', 'parmesan', 'to', 'taste'
    ]

    expected_matrix = [
        [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    ]

    # Проверяем имена признаков в точном порядке
    assert vectorizer.get_feature_names() == expected_features

    # Проверяем матрицу счётчиков
    assert count_matrix == expected_matrix

def test_feature_names_length_matches_matrix_columns():
    corpus = [
        'a b c',
        'b c d'
    ]
    vec = CountVectorizer()
    matrix = vec.fit_transform(corpus)
    features = vec.get_feature_names()
    # число столбцов матрицы должно совпадать с числом фич
    assert all(len(row) == len(features) for row in matrix)
    