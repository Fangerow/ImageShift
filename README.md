# ImageShift  

## Описание реализованных подходов  
### Смещение изображения на 3.5 писеля  
Смещение изображения было реализовано при помощи аффинного преобразования из модуля cv2. 
В данном случае, матрица преобразования задает сдвиг на 3.5 пикселя по обеим осям (x и y).  

Так как сдвиг задан не целым числом пикселей, новые пиксели могут находиться между исходными пикселями изображения.   
Чтобы определить цвет новых пикселей, используется метод бикубической интерполяции.

### Используемые архитектуры  
В ходе работы над проектом были опробованы различные подходы для обучения модели на основе исходного датасета.  
В числе исследованных методов были модели   GAN(Convolutional GAN, или ConvGAN), а также вариационные автоэнкодеры.  
Однако, эти подходы не дали желаемого результата; вероятно, для таких сложных моделей исходный размер датасета не является достаточным. 


Были также рассмотрены две более специализированные архитектуры.  
Первая из них - это Spatial Transformer Networks (STN). Такая архитектура была выбрана, так как она яявляется специализированной для задач преобразования входных данных.  
На данной архитетуре (CNN + Афинные преобразования) был получен удовлетворительный результат.

Второй подход, который показался перспективным и оптимальным, включает использование единичной свертки.  
Несмотря на то, что этот метод  не демонстрирует ожидаемого качества работы (на текущий момернт),  
его неудача может быть связана скорее с неточностями в пайплайне и подбором гиперпараметров, а не с принципиальной непригодностью идеи.  
Основной концепцией этого подхода является использование матрицы сдвига, которую модель оптимизирует в процессе обучения, сходясь к "правильной".

### Оценка ошибки
Мною былро рассмотрено несколько способов, включая самых простых и очевидных (l1 и l2 норма разница между матрицами соответствующий изображений).  
Однако наиболее стабильным критерием на произвольных изображениях и диапазонах значений сдвига показал себя оптический поток.  
Данная метрика была выбрана по двум причинам:  
1.  Оптический поток способен обнаруживать смещения между изображениями, которые меньше одного пикселя. Это особенно полезно в вашем случае, так как мы исследуете небольшой сдвиг, на 3.5 пикселя.
2.  В отличие от некоторых других метрик ошибки, оптический поток учитывает пространственную структуру изображения. Это означает, что он не просто смотрит на общую разницу между двумя изображениями, но также учитывает, как эта разница распределена по изображению.

### Результаты  
Для оптимальной модели удалось достичь разница порядка 1e-6 между сдвигом при помощи метода opencv (изображения в Data/dataset/dist) и сдвигом в результате работы нейросети.  
(заметка: типовой поряд величины между изображением из dist и scr составялет 1e-4, на "плохих" моделях выходит такой же порядок или даже выше).
Таким образом, подход оптимален, так как целевой эффект очень трудно заметить "на глаз".
