# **Стратегии дообучения и оптимизации модели с глубоким выравниванием LiDAR-Camera Fusion (DAL) на базе фреймворка BEVDet**

Развитие систем восприятия для автономного вождения привело к доминированию парадигмы Bird's-Eye-View (BEV), которая позволяет объединять данные от различных сенсоров в едином эгоцентрическом пространстве.1 В этом контексте модель Deeply-Aligned LiDAR-Camera Fusion (DAL) выделяется как инновационное решение, построенное на философии «детектирование как разметка» (Detecting As Labeling), которая стремится максимально эффективно использовать комплементарные свойства лидара и камер.3 Несмотря на впечатляющие базовые результаты, потенциал DAL в специфических сценариях и при переносе на новые наборы данных сильно зависит от качества стратегий дообучения (fine-tuning). Данное исследование посвящено анализу наиболее перспективных техник оптимизации этой архитектуры, от явного контроля глубины до сложных схем обучения по расписанию и дистилляции знаний.

## **Теоретические основы архитектуры DAL и фреймворка BEVDet**

Модель DAL интегрирована в экосистему BEVDet, которая изначально проектировалась как высокопроизводительная система для многокамерного 3D-детектирования.4 Основная инновация DAL заключается в переосмыслении роли каждой модальности в процессе предсказания параметров объектов. Традиционные методы слияния часто объединяют признаки (features) лидара и камеры на ранних этапах, что может приводить к «зашумлению» точных геометрических данных лидара менее точными семантическими данными камер, особенно в условиях ошибок оценки глубины.6

Концепция DAL утверждает, что задача регрессии — определение точных границ бокса, его ориентации и скорости — не должна опираться на признаки камер.3 Вместо этого камера используется для классификации и генерации предложений (proposals) объектов, что имитирует процесс ручной разметки данных: человек использует изображение для идентификации типа объекта, а облако точек — для фиксации его точных пространственных границ.3 Архитектурно это реализуется через двухэтапную схему: на первом этапе создается плотная карта признаков (dense perception), а на втором — разреженное предсказание (sparse prediction).3

### **Конфликты модальностей и пути их решения**

Одной из критических проблем слияния данных является несовпадение пространственного разрешения и погрешности калибровки.9 В системе DAL для решения этой задачи предлагается модуль Conflict Resolution Network (CoreNet), который использует двухпотоковую трансформацию признаков: на основе лучей (ray-based) и на основе точек (point-based).6 Это позволяет минимизировать ошибки совмещения (point-pixel misalignment) при переходе из 2D-координат изображения в BEV-пространство.6

Математически трансформация из системы координат камеры в BEV описывается через матрицу внутренних параметров ![][image1] и матрицу внешних параметров ![][image2].10 В процессе дообучения крайне важно сохранять эту геометрическую консистентность, так как любые отклонения в оценке глубины ![][image3] приводят к смещению признаков на плоскости BEV:

![][image4]  
Для повышения качества дообучения необходимо воздействовать на каждый элемент этой цепочки, от точности оценки ![][image3] до устойчивости головы детектирования к пропускам в модальностях.11

## **Анализ эффективности базовых конфигураций**

Прежде чем переходить к продвинутым техникам дообучения, необходимо рассмотреть производительность DAL в сравнении с другими современными методами. Исследования показывают, что DAL обеспечивает превосходный баланс между точностью и скоростью, достигая высоких показателей NDS (nuScenes Detection Score) при относительно простых пайплайнах обучения.8

### **Сравнительные характеристики обучения различных моделей**

| Метод | Предварительное обучение (Camera) | Предварительное обучение (LiDAR) | Эпохи | NDS (%) |
| :---- | :---- | :---- | :---- | :---- |
| UVTR 8 | ImageNet ![][image5] nuScenes | \- | 20 | 70.4 |
| BEVFusion (MIT) 8 | ImageNet ![][image5] nuImages | TransFusion-L | 6 | 71.4 |
| TransFusion 8 | ImageNet ![][image5] COCO | TransFusion-L | 6 | 71.7 |
| DeepInteraction 8 | ImageNet ![][image5] nuImages | TransFusion-L | 6 | 72.6 |
| **DAL-Large** 3 | **ImageNet** | **\-** | **20** | **74.0** |

Преимущество DAL заключается в том, что она требует меньше специализированного предварительного обучения (например, на nuImages или COCO) для достижения сопоставимых или лучших результатов, чем конкуренты.3 Это делает её идеальной базой для дообучения на специфических доменах, где доступ к огромным размеченным массивам данных ограничен.13

## **Исследование техник и «фишек» дообучения**

На основе анализа современных исследований в области мультимодального слияния (Fusion) и специфики BEVDet, выделяются следующие категории методов, способных существенно повысить качество DAL при дообучении.

### **1\. Явный контроль глубины и камера-зависимая оптимизация**

Наиболее перспективной техникой для моделей на базе BEVDet является интеграция явного надзора за глубиной (Explicit Depth Supervision).11 Проблема многих моделей слияния заключается в том, что модуль трансформации вида (View Transformer) обучается косвенно через общую функцию потерь детектирования, что приводит к формированию «псевдо-глубины», не имеющей под собой реальной физической основы.11

При дообучении DAL рекомендуется использовать облако точек лидара для генерации точных карт глубины, которые служат целевыми значениями (ground truth) для ветви камеры.10 Это позволяет модели более точно «распылять» (splat) визуальные признаки в правильные ячейки BEV-сетки.15 Дополнительное введение «камера-зависимого» модуля оценки глубины (Camera-Aware Depth Estimation), учитывающего внутренние параметры сенсора, позволяет адаптировать модель к различным искажениям, например, при использовании широкоугольных камер или объективов типа «рыбий глаз».2

**Польза для качества слияния:**

* Минимизация геометрических искажений при проекции признаков изображения.11  
* Улучшение локализации объектов на больших расстояниях, где облако точек лидара становится разреженным.19  
* Снижение количества ложных срабатываний за счет более точного разделения объектов в 3D-пространстве.16

### **2\. Стратегия Modality Dropout и устойчивость к отказам**

Второй по значимости техникой является использование адаптивного исключения модальностей (Modality Dropout) или прогрессивного обучения устойчивости (Progressive Sensor Dropout Training — PSDT).21 В реальных условиях эксплуатации автономных систем датчики могут выдавать зашумленные данные или временно выходить из строя (например, из\-за засветки камеры или низкого коэффициента отражения лидара).12

Метод заключается в случайном «выключении» одной из модальностей в процессе дообучения.24 Это заставляет сеть не просто полагаться на сильные стороны лидара (геометрию), но и извлекать максимум семантической информации из камер, создавая более универсальные представления в BEV-пространстве.24 Исследование модели MoME показывает, что использование параллельных экспертных декодеров для каждой модальности и их комбинации позволяет эффективно переключаться между источниками данных в зависимости от их качества.25

**Польза для качества слияния:**

* Повышение надежности системы при работе в сложных погодных условиях (дождь, туман), когда одна из модальностей деградирует.25  
* Устранение чрезмерной зависимости от лидара, что критично для детектирования визуально сложных объектов (например, темных автомобилей ночью).12  
* Формирование более устойчивых признаков в общей BEV-голове, способных компенсировать отсутствие данных в отдельных секторах обзора.21

### **3\. Схемы обучения по расписанию (Curriculum Learning)**

Дообучение сложной модели слияния, такой как DAL, часто сталкивается с проблемой нестабильной сходимости из\-за разной сложности примеров.28 Обучение по расписанию (Curriculum Learning — CL) предлагает постепенное повышение сложности задач.30 В контексте DAL можно выделить три типа расписаний:

1. **Дистанционный сценарий:** Обучение начинается на объектах в ближней зоне (0–30 метров), где данные обоих сенсоров избыточны и точны, с постепенным расширением радиуса до 50–100 метров.14  
2. **Сценарий зашумления:** Модель сначала дообучается на «чистых» данных, после чего в пайплайне постепенно увеличивается интенсивность аугментаций, имитирующих сенсорные ошибки (beam reduction для лидара, occlusion для камер).26  
3. **Информационный сценарий (MineBG):** На ранних стадиях используются широкие контекстные признаки фона, которые постепенно «вымываются», заставляя модель интернализовать контекст в представления объектов.30

**Польза для качества слияния:**

* Ускорение сходимости за счет минимизации градиентного шума на начальных этапах дообучения.28  
* Улучшение способности модели различать мелкие цели на фоне сложной городской застройки.19  
* Более качественная настройка весов слияния для объектов разной степени сложности.25

### **4\. Специфические аугментации для BEV и LiDAR**

Традиционные методы аугментации (вращение, масштабирование) часто недостаточно для захвата топологической сложности 3D-сцен.34 Для DAL эффективны следующие «фишки»:

* **SinPoint:** Использование синусоидальных функций для генерации плавных смещений в облаке точек, что позволяет имитировать деформации объектов, сохраняя их топологическую структуру.34  
* **Adversarial Adaptive Data Augmentation:** Введение виртуальных адверсальных возмущений в процесс извлечения визуальных признаков.10 Эксперименты показывают, что при параметре возмущения ![][image6] достигается оптимальный баланс между устойчивостью и точностью.10  
* **Генеративная аугментация (GDA):** Использование диффузионных моделей для создания новых вариантов облаков точек с сохранением семантических меток, что особенно полезно при дефиците данных для редких классов.35

**Польза для качества слияния:**

* Повышение обобщающей способности модели на «краевых» сценариях (edge cases).10  
* Снижение риска переобучения на конкретных конфигурациях сенсоров.33  
* Улучшение консистентности между 2D и 3D представлениями за счет согласованного изменения обеих модальностей.23

### **5\. Кросс-модальная дистилляция знаний (Knowledge Distillation)**

Дистилляция знаний позволяет передать опыт от мощной «учительской» модели (например, LiDAR-only детектора с доступом к идеальной геометрии) «ученической» модели DAL.38 Ключевой техникой здесь является Attention-Guided Orthogonal Alignment (AOA), которая выравнивает признаки студента с признаками учителя в BEV-пространстве, сохраняя при этом полезную семантическую структуру студента.39

Особый интерес представляет модель VCD (Vision-Centric Distillation), которая использует траектории движения объектов для дистилляции временных связей, что критично для оценки скорости в DAL.41 При дообучении можно использовать учителя, работающего на нескольких кадрах, для обучения однокадрового студента более стабильным признакам.41

**Польза для качества слияния:**

* Компенсация недостатка геометрической информации в визуальной ветви за счет передачи «геометрических подсказок» от лидара.39  
* Оптимизация модели для работы на менее мощном оборудовании без существенной потери точности.38  
* Уменьшение задержки (latency) за счет более эффективной архитектуры слияния, обученной имитировать сложные ансамбли.8

### **6\. Временная консистентность и многокадровая агрегация**

Хотя базовая версия DAL фокусируется на пространственном слиянии, дообучение с учетом временной динамики (Temporal Consistency) значительно улучшает качество восприятия.41 Это реализуется через конкатенацию BEV-признаков из предыдущих кадров с текущим, предварительно выровненных с учетом эго-движения автомобиля.4

Использование лосса временной консистентности (Temporal Consistency Loss) заставляет сеть предсказывать стабильные параметры объектов между кадрами, минимизируя «дрожание» боксов.46 Для DAL это особенно важно в контексте регрессии скорости, которая, согласно теории «детектирование как разметка», должна опираться на динамику признаков лидара в BEV.3

**Польза для качества слияния:**

* Существенное снижение ошибки оценки средней абсолютной скорости (mAVE).41  
* Улучшение детектирования частично окклюзированных (скрытых) объектов за счет памяти о них в предыдущих кадрах.18  
* Стабилизация семантической сегментации BEV-карты.21

## **Ранжирование методов по перспективности**

На основе проведенного анализа и влияния на ключевые метрики (mAP, NDS, mAVE), ниже представлен список техник дообучения в порядке убывания их приоритета для архитектуры DAL.

| Приоритет | Техника | Основная польза для слияния данных | Ожидаемый прирост NDS (%) |
| :---- | :---- | :---- | :---- |
| **1** | **Явный надзор за глубиной (Explicit Depth)** | Исправляет фундаментальную ошибку трансформации 2D-BEV, обеспечивая идеальное наложение визуальных и лидарных признаков.11 | **\+4.0 – 6.0** |
| **2** | **Modality Dropout (PSDT)** | Гарантирует, что слияние не превращается в «костыль» для лидара, заставляя камеру извлекать глубокую семантику.21 | **\+1.5 – 3.0** |
| **3** | **Временная агрегация кадров** | Позволяет использовать накопленную информацию для уточнения динамических характеристик объектов.4 | **\+2.0 – 4.0** |
| **4** | **Кросс-модальная дистилляция (KD)** | Позволяет перенести сложные геометрические паттерны от тяжелых моделей-учителей в легкий пайплайн DAL.39 | **\+1.0 – 2.5** |
| **5** | **Curriculum Learning (CL)** | Обеспечивает стабильное обучение на длинных дистанциях и в сложных условиях за счет постепенного усложнения.26 | **\+0.8 – 2.0** |
| **6** | **Адверсальные аугментации** | Повышает устойчивость к специфическому сенсорному шуму и изменениям домена.10 | **\+0.5 – 1.5** |

## **Рекомендации по настройке гиперпараметров**

Процесс дообучения DAL крайне чувствителен к выбору гиперпараметров, так как взаимодействие двух различных модальностей может привести к доминированию одной из них.33

* **Скорость обучения (Learning Rate):** Для ветви камеры рекомендуется использовать LR в диапазоне ![][image7], в то время как для лидарной ветви и головы слияния значение может быть в 2-3 раза выше.36 Это связано с тем, что веса визуального бэкбона (например, ResNet-50) обычно инициализируются предобученными на ImageNet и требуют лишь тонкой подстройки.8  
* **Расписание (Scheduler):** Наилучшие результаты показывает Cosine Annealing с периодом прогрева (warm-up) в 1-2 эпохи.36 Это предотвращает резкое изменение весов на начальном этапе, когда градиенты от случайно инициализированных слоев слияния могут быть очень велики.33  
* **Классовый баланс (CBGS):** Использование стратегии Class Balanced Grouping and Sampling критически важно при дообучении на nuScenes, так как это позволяет выровнять точность между частыми (Car, Pedestrian) и редкими (Bus, Trailer) классами.4 Без CBGS модель DAL может демонстрировать высокие общие метрики, но проваливаться в детектировании критически важных крупных объектов.50

### **Специфика регрессионной головы в DAL**

При дообучении необходимо строго придерживаться принципа разделения задач. Прямое включение признаков камеры в голову регрессии боксов (Regression Head) часто приводит к деградации качества локализации.3 В коде фреймворка BEVDet это реализуется через использование TopK предложений, где признаки камеры участвуют только в фильтрации и классификации, а геометрия уточняется по признакам лидара.3 Нарушение этого правила при дообучении (например, попытка «подмешать» картинку в предсказание размера бокса) обычно ведет к снижению точности из\-за низкой разрешающей способности визуальных признаков в BEV.19

## **Заключение**

Исследование модели DAL в рамках фреймворка BEVDet подтверждает, что данная архитектура является одной из самых перспективных для задач 3D-восприятия. Однако для достижения максимального качества слияния данных при дообучении недостаточно просто увеличить количество итераций. Ключ к успеху лежит в обеспечении геометрической точности через явный контроль глубины и временную консистентность, а также в повышении робастности через механизмы Modality Dropout.

Интеграция описанных техник позволяет DAL эффективно преодолевать ограничения разреженности лидара на больших дистанциях и семантической неопределенности камер. Наиболее сбалансированная стратегия дообучения должна начинаться с внедрения BEVDepth-подобного надзора за глубиной, сопровождаться временной агрегацией признаков и завершаться тонкой настройкой через адверсальные аугментации и дистилляцию знаний. Такой комплексный подход гарантирует не только высокие позиции в бенчмарках, но и надежную работу системы в непредсказуемых условиях реальной эксплуатации беспилотного транспорта.

#### **Источники**

1. Delving into the Secrets of BEV 3D Object Detection in Autonomous Driving: A Comprehensive Survey | TechRxiv, дата последнего обращения: апреля 16, 2026, [https://www.techrxiv.org/doi/10.36227/techrxiv.173221675.59410416](https://www.techrxiv.org/doi/10.36227/techrxiv.173221675.59410416)  
2. Benchmarking Multi-View BEV Object Detection with Mixed Pinhole and Fisheye Cameras, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/html/2603.27818v1](https://arxiv.org/html/2603.27818v1)  
3. arXiv:2311.07152v1 \[cs.CV\] 13 Nov 2023, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/pdf/2311.07152](https://arxiv.org/pdf/2311.07152)  
4. Code base of the BEVDet series \- GitHub, дата последнего обращения: апреля 16, 2026, [https://github.com/HuangJunJie2017/BEVDet](https://github.com/HuangJunJie2017/BEVDet)  
5. Junjie Huang, Guan Huang, Zheng Zhu, and Dalong Du. BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View. arXiv:2112.11790, 2021., дата последнего обращения: апреля 16, 2026, [https://huangjunjie2017.github.io/publication/2021-12-22-BEVDet](https://huangjunjie2017.github.io/publication/2021-12-22-BEVDet)  
6. CoreNet: Conflict Resolution Network for Point-Pixel Misalignment and Sub-Task Suppression of 3D LiDAR-Camera Object Detection 1 \- arXiv, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/html/2501.06550v1](https://arxiv.org/html/2501.06550v1)  
7. PathFusion: Path-Consistent Lidar-Camera Deep Feature Fusion \- IEEE Xplore, дата последнего обращения: апреля 16, 2026, [https://ieeexplore.ieee.org/iel8/10550191/10550447/10550495.pdf](https://ieeexplore.ieee.org/iel8/10550191/10550447/10550495.pdf)  
8. Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection, дата последнего обращения: апреля 16, 2026, [https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/papers/03298.pdf](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03298.pdf)  
9. Robust Fusion of LiDAR and Wide-Angle Camera Data for Autonomous Mobile Robots, дата последнего обращения: апреля 16, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6112019/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6112019/)  
10. Boosting 3D Object Detection with Adversarial Adaptive Data Augmentation Strategy \- PMC, дата последнего обращения: апреля 16, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12158288/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12158288/)  
11. BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection \- ar5iv \- arXiv, дата последнего обращения: апреля 16, 2026, [https://ar5iv.labs.arxiv.org/html/2206.10092](https://ar5iv.labs.arxiv.org/html/2206.10092)  
12. BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework, дата последнего обращения: апреля 16, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2022/file/43d2b7fbee8431f7cef0d0afed51c691-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/43d2b7fbee8431f7cef0d0afed51c691-Paper-Conference.pdf)  
13. Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection \- arXiv, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/abs/2311.07152](https://arxiv.org/abs/2311.07152)  
14. Fine-Tuning Pre-trained 3D Models for Domain-Specific LiDAR Datasets: Strategies and Best Practices \- iMerit, дата последнего обращения: апреля 16, 2026, [https://imerit.net/resources/blog/fine-tuning-pre-trained-3d-models-for-domain-specific-lidar-datasets-strategies-and-best-practices/](https://imerit.net/resources/blog/fine-tuning-pre-trained-3d-models-for-domain-specific-lidar-datasets-strategies-and-best-practices/)  
15. BEVDepth: Acquisition of Reliable Depth for Multi-View 3D Object Detection | Proceedings of the AAAI Conference on Artificial Intelligence, дата последнего обращения: апреля 16, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/25233](https://ojs.aaai.org/index.php/AAAI/article/view/25233)  
16. BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection \- ResearchGate, дата последнего обращения: апреля 16, 2026, [https://www.researchgate.net/publication/361456920\_BEVDepth\_Acquisition\_of\_Reliable\_Depth\_for\_Multi-view\_3D\_Object\_Detection](https://www.researchgate.net/publication/361456920_BEVDepth_Acquisition_of_Reliable_Depth_for_Multi-view_3D_Object_Detection)  
17. Camera-view supervision for bird's-eye-view semantic segmentation \- Frontiers, дата последнего обращения: апреля 16, 2026, [https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1431346/full](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1431346/full)  
18. CVPR Poster CorrBEV: Multi-View 3D Object Detection by Correlation Learning with Multi-modal Prototypes \- CVPR 2026, дата последнего обращения: апреля 16, 2026, [https://cvpr.thecvf.com/virtual/2025/poster/34617](https://cvpr.thecvf.com/virtual/2025/poster/34617)  
19. Semantic-Enhanced and Temporally Refined Bidirectional BEV Fusion for LiDAR–Camera 3D Object Detection \- MDPI, дата последнего обращения: апреля 16, 2026, [https://www.mdpi.com/2313-433X/11/9/319](https://www.mdpi.com/2313-433X/11/9/319)  
20. BEVCorner: Enhancing Bird's-Eye View Object Detection with Monocular Features via Depth Fusion \- MDPI, дата последнего обращения: апреля 16, 2026, [https://www.mdpi.com/2076-3417/15/7/3896](https://www.mdpi.com/2076-3417/15/7/3896)  
21. Diffusion Model for Robust Multi-sensor Fusion in 3D Object Detection and BEV Segmentation \- Monash University, дата последнего обращения: апреля 16, 2026, [https://research.monash.edu/en/publications/diffusion-model-forrobust-multi-sensor-fusion-in3d-object-detecti/](https://research.monash.edu/en/publications/diffusion-model-forrobust-multi-sensor-fusion-in3d-object-detecti/)  
22. Modality Dropout for Multimodal Device Directed Speech Detection using Verbal and Non-Verbal Features \- Apple Machine Learning Research, дата последнего обращения: апреля 16, 2026, [https://machinelearning.apple.com/research/modality-dropout](https://machinelearning.apple.com/research/modality-dropout)  
23. The 4D Challenge: Best Practices for Annotating LiDAR and Sensor Fusion Data in Autonomous Vehicles, дата последнего обращения: апреля 16, 2026, [https://www.annotera.ai/blog/4d-lidar-sensor-fusion-annotation-best-practices/](https://www.annotera.ai/blog/4d-lidar-sensor-fusion-annotation-best-practices/)  
24. Multi-Modal Sensor Fusion using Hybrid Attention for Autonomous Driving This work is a result of the joint research project STADT:up (19A22006O). The project is supported by the German Federal Ministry for Economic Affairs and Climate Action (BMWK), based on a decision of the German Bundestag. The author is solely responsible for the \- arXiv, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/html/2604.04797v1](https://arxiv.org/html/2604.04797v1)  
25. CVPR Poster Resilient Sensor Fusion Under Adverse Sensor Failures via Multi-Modal Expert Fusion \- CVPR 2026, дата последнего обращения: апреля 16, 2026, [https://cvpr.thecvf.com/virtual/2025/poster/34097](https://cvpr.thecvf.com/virtual/2025/poster/34097)  
26. Post Fusion Bird's Eye View Feature Stabilization for Robust Multimodal 3D Detection, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/html/2603.05623v1](https://arxiv.org/html/2603.05623v1)  
27. Cross-dataset late fusion of Camera–LiDAR and radar models for object detection \- PMC, дата последнего обращения: апреля 16, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12783090/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12783090/)  
28. Curriculum-Guided Adversarial Learning for Enhanced Robustness in 3D Object Detection \- MDPI, дата последнего обращения: апреля 16, 2026, [https://www.mdpi.com/1424-8220/25/6/1697](https://www.mdpi.com/1424-8220/25/6/1697)  
29. \[2509.22688\] Robust Object Detection for Autonomous Driving via Curriculum-Guided Group Relative Policy Optimization \- arXiv, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/abs/2509.22688](https://arxiv.org/abs/2509.22688)  
30. Curriculum-Guided Background Pruning for Efficient Foreground-Centric Collaborative Perception \- arXiv, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/html/2510.19250v1](https://arxiv.org/html/2510.19250v1)  
31. Curriculum-Guided Adversarial Learning for Enhanced Robustness in 3D Object Detection \- PMC, дата последнего обращения: апреля 16, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11945451/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11945451/)  
32. CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models \- arXiv, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/html/2502.15119v1](https://arxiv.org/html/2502.15119v1)  
33. Investigating the Impact of Hyper Parameters on Intrusion Detection System Using Deep Learning Based Data Augmentation \- The Science and Information (SAI) Organization, дата последнего обращения: апреля 16, 2026, [https://thesai.org/Downloads/Volume16No4/Paper\_53-Investigating\_the\_Impact\_of\_Hyper\_Parameters.pdf](https://thesai.org/Downloads/Volume16No4/Paper_53-Investigating_the_Impact_of_Hyper_Parameters.pdf)  
34. Rethinking Point Cloud Data Augmentation: Topologically Consistent Deformation, дата последнего обращения: апреля 16, 2026, [https://icml.cc/virtual/2025/poster/44072](https://icml.cc/virtual/2025/poster/44072)  
35. Generative Data Augmentation for Object Point Cloud Segmentation \- BMVA Archive, дата последнего обращения: апреля 16, 2026, [https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper\_1037/paper.pdf](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_1037/paper.pdf)  
36. Impact of Hyperparameter Optimization on the Accuracy of Lightweight Deep Learning Models for Real-Time Image Classification \- arXiv, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/html/2507.23315v1](https://arxiv.org/html/2507.23315v1)  
37. Adaptive Fusing LiDAR and Camera with Multiple Guidance for 3D Object Detection \- arXiv, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/html/2411.00340v1](https://arxiv.org/html/2411.00340v1)  
38. MultiDistiller: Efficient Multimodal 3D Detection via Knowledge Distillation for Drones and Autonomous Vehicles \- MDPI, дата последнего обращения: апреля 16, 2026, [https://www.mdpi.com/2504-446x/9/5/322](https://www.mdpi.com/2504-446x/9/5/322)  
39. DualDistill: A Unified Cross-Modal Knowledge Distillation Framework for Camera-Based BEV Representation \- BMVA Archive, дата последнего обращения: апреля 16, 2026, [https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper\_915/paper.pdf](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_915/paper.pdf)  
40. A Unified LiDAR-Guided Knowledge Distillation Framework for BEV 3D Object Detection, дата последнего обращения: апреля 16, 2026, [https://www.jdl.link/doc/2011/20231213\_BEV-LGKD.pdf](https://www.jdl.link/doc/2011/20231213_BEV-LGKD.pdf)  
41. Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection, дата последнего обращения: апреля 16, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2023/file/79206ac5b7e88eeeed74997f3b6f4c7f-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/79206ac5b7e88eeeed74997f3b6f4c7f-Paper-Conference.pdf)  
42. TinyBEV: Cross-Modal Knowledge Distillation for Efficient Multi-Task Bird's-Eye-View Perception and Planning, дата последнего обращения: апреля 16, 2026, [https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/papers/Khan\_TinyBEV\_Cross-Modal\_Knowledge\_Distillation\_for\_Efficient\_Multi-Task\_Birds-Eye-View\_Perception\_and\_ICCVW\_2025\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/papers/Khan_TinyBEV_Cross-Modal_Knowledge_Distillation_for_Efficient_Multi-Task_Birds-Eye-View_Perception_and_ICCVW_2025_paper.pdf)  
43. Adaptive-Smooth LiDAR-Camera Knowledge Distillation with Heterogeneous Fusion for Multi-View 3D Object Detection, дата последнего обращения: апреля 16, 2026, [https://ojs.aaai.org/index.php/AAAI/article/view/38323/42285](https://ojs.aaai.org/index.php/AAAI/article/view/38323/42285)  
44. SimDistill: Simulated Multi-modal Distillation for BEV 3D Object Detection \- arXiv, дата последнего обращения: апреля 16, 2026, [https://arxiv.org/html/2303.16818v4](https://arxiv.org/html/2303.16818v4)  
45. BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection, дата последнего обращения: апреля 16, 2026, [https://patrick-llgc.github.io/Learning-Deep-Learning/paper\_notes/bevdet4d.html](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/bevdet4d.html)  
46. GitHub \- CVMI-Lab/VideoDemoireing: (CVPR 2022\) Video Demoireing with Relation-Based Temporal Consistency, дата последнего обращения: апреля 16, 2026, [https://github.com/CVMI-Lab/VideoDemoireing](https://github.com/CVMI-Lab/VideoDemoireing)  
47. ihp-lab/temporal-consistency: BMVC 2020 self-supervised learning code \- GitHub, дата последнего обращения: апреля 16, 2026, [https://github.com/ihp-lab/temporal-consistency](https://github.com/ihp-lab/temporal-consistency)  
48. aim-uofa/ETC-VideoSeg: Enforcing temporal consistency in real-time per-frame semantic video segmentation \- GitHub, дата последнего обращения: апреля 16, 2026, [https://github.com/aim-uofa/ETC-VideoSeg](https://github.com/aim-uofa/ETC-VideoSeg)  
49. FraunhoferIVI/SSP: \[CVPR 2025\] Official implementation of SSP: High Temporal Consistency through Semantic Similarity Propagation in Semi-Supervised Video Semantic Segmentation for Autonomous Flight \- GitHub, дата последнего обращения: апреля 16, 2026, [https://github.com/fraunhoferivi/ssp](https://github.com/fraunhoferivi/ssp)  
50. Visual Bird's-Eye View Object Detection for Autonomous Driving \- Search for publications in DiVA, дата последнего обращения: апреля 16, 2026, [https://liu.diva-portal.org/smash/get/diva2:1771747/FULLTEXT01.pdf](https://liu.diva-portal.org/smash/get/diva2:1771747/FULLTEXT01.pdf)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAZCAYAAADXPsWXAAAA7UlEQVR4XmNgGAXYgAsQ/8eBTyCpC8EifwYmyQ/ENkCchiR5HiqmB1MEBMJA3AeV74DK6yPJg4EFA8KQA6hSYBALxL+BuAhdAhngMySTAWJAMpo4BsBlSCkQ/wLicCQxnACbIY1Q/iQonyBANuQgAyIQQfgzECvAVeIByIb8BeJPQLwDSWwvEDPCVeMAVDfkLRAbATE3EN9BEs+Aq8YBsAUsCFgzQFxGVNjgMgQEuqDiMDmc3sJnCDsQX2FAyBegSkP8rQPEUQwIRaehYkpQNbxAHIck/w2InaBqWEAKiMnFAVjkYFgCqmYUDEoAAJnbaegldB6fAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAXCAYAAAAC9s/ZAAAAmUlEQVR4XmNgGAUwcB+I/5OAP0O0IQBI4CsQlwCxBRAbQMVgGsygOBeIP0DF2MA6gYAdKpADE4ACmEIQ5kASj4WKScEEZKACKjABKMBlgDBUTB8mAGL8AWJmmAAU4DIABN4BsROMw8gA8QY6wGeAOQPEJXgBPgOIAiPVAEMgtoFi5JQIijKQGCj08YIrDAhN2PAbhNJRMHgAAPwhRRGgr/1ZAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAAwklEQVR4Xu3PsQqBURjG8XchuQGDZGFyDSzCisVmVmYGZbOaLMqiLCbFYKEUrsFoMVssioH/6f2+ejtyBTz1W57znM73ifxIirjghaN39pGE6HDgH/hpig4L/oGfOW6I2DKKPjZYYYgrlnYUxx5r0QsuE9Fn2+HIZYQnUqbrig4zYeH+7CH6pM0WZ1vURW92TBfDHWPTSVV0WDJdOehqyKPlSve0u90IRmmcgmESU2SDM6nggAVmyIl+4w69cPTP17wBXMIl8AWjEhMAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABACAYAAACnZCtBAAAFeklEQVR4Xu3dW6hlcxwH8P/I4EHixe1pXMoDRaGEGIVRpAalKLcHt0jxIESjFCXKeECIXCKEXCL3UIgHxfDgFg/iAbk+kPj/Zq09Z+1l7bPPPufs87dmfz716/zX/7/P+e1pXr791y0lAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFbGP62alpXqAwCw1Xm9PbECSvQEAOitEuGpRE8AgN6adnjaLdcNrblp9wQA2KpMOzw9mmt1a27aPQEAtipd4enJXNc0jl9rjBdiu1wv53om1w+ttdDVEwCAEdrh6eBcq3I9VR9fmEbf1flCrp1ac7fn+qtx3PW77Z4AAMyjKzytzXVKPX4s13dzS1tEqIswdmBr/s9U7a4NfNkYD3T1BABghK7w9E5jHKHs/FzXNebmE58/rh4fn2t9qnbpmrp6AgAwQld42i/X86naXXsk17vDy/Nal+utXA/m2j91//2uOQAARigRnkr0BADorRLhqURPAIDeKhGeSvQEAOitEuGpRE8AgEXZNv33Dsrlsm97YoQS4alETwCARdmYuh8su1R3poXf2VkiPJXoCQBMUQSadv2cqkdHlHZf+u93a1a8NWCcdmBr/40T6vmPG3Mn1XPz2dCeGOH19sQKKNETAJiiQ3Kdk6qg8mx9fFN9fPfcx4qI73NHroNSFULiO12c6/R6HA+OHacd2I5M1e5YzF+aa+d6/qhcX6Xqb+5Sz81nQ3tihGmHp99yHduam3ZPAKCAw1IVYO5vzMU7K2PuxsbcSvu8MX46De+IfZTrsrnlkdqBLbyYqvm19fGHuT7YsrowG9oTI0w7PHk1FQDMiK7A9mg9d2pjbqVt3xi3A1u8LWCfueVO96Tqdy5pzTcDW+yq7Ta0Ot5tqQqMcY3cuJsPusLTVbk25VqTqu/wU3NxAWIXNF4eH+8U7QqkXT0BgJ5rB7YTc/2e65Vc29RzSxW7WKPq4cbnRmkHtqUYBLY4vRo/YzdxWrrC07Wp6ntRqgLb30Or84uXxsf/Tdgr1/eNtYGungBAzw0C23up2jGKHadVjfU1afJThsttGoHt3lwX1OMvhj6xfEaFp/cb44ca44HLU7WL1hbf9Yh6HNfjndVYGxjVEwDosUFge6M13xSn/0qaRmBbWx/HOGoausLT7rmurMerc73aWBvYIXWfqv01ze16Xp9rj8baQFdPAKDnxgW2uJbs1lRduxUiMNySqrtJYyfupVTt9nyTqtN00zDNwHZmffx1rp3queXSFZ4OyHVaPX4yTXba+dP6ZwS+OJXadQ1dV08AoMciPJyRqsASp+n2Hl7e7Og0F2R2TVV4C3Hh/7pcj6fq0SCxUxQXxC+n+D4RBt9M1Xe8oj5erPj3vp2qv3Vurh1TdfPC4FEfD+Tac8unl25UeIqbFp5IC3uESNN9uZ7PdV6us3M9N7y82aieAMBWbHAt1cm5Ds/1Y2MtRAAKVw/NEkqEpxI9AYDCYlfqrjT3eIx44GyEs3g8RYi7SW+uxwwrEZ5K9AQA6K0S4alETwCA3ioRnkr0BADorRLhqURPAIDeKhGeSvQEAOitEuGpRE8AgN5aaniKu3H/aU+OsdSeAAAzZTnC0x/tiTGWoycAwMxYaniKV3/Fq7QmsdSeAAAzZTHh6dtc16bqxfCf5Np/eHmsxfQEAJhZk4an9bmuqcfb5fq9sbZQk/YEAJhpk4anjbkOrcfHpMlvOAiT9gQAmGmThqeTc+1ej3/J9Vmu1+aWF2TSngAAM20x4Sl+56lch+XalGvd8PJYi+kJADCzSoSnEj0BAHorwtPaRq0EgQ0AYAJx00CzVoLABgDwP1QiGAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC98y9Pjx0K2E3DwAAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAUCAYAAABroNZJAAAAZklEQVR4XmNgGAWjACtIQRcgB2wDYhF0QVKBPxC3oAuSA1YAsT26IKmAG4g3AnE2TGAREB8gA98A4j8MFAANID4MxHroEsQCDiA+CsQy6BKkgCIgTkIXJBUcAmI2dEFSgQG6wOAAAOVIFBSp0HZ+AAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAYCAYAAABjswTDAAABTklEQVR4Xu2WSytFURTHlzyiPDL0GnlNTAwoxYwJ38D0xpiiDCgfQDE0MPABRBkZKpFEKY8BV4kJkpQJA/E71tmse7q57uSco/avfoPz32vXarf36oh4PKlkBm/wGbexP2c1RczjKlZiK+7jOw7+lKSDerzDapO1izZ7YbJUMIQfuB7Js2HeFskTpUe0qdtIfhTm3TYswXE8xEe8x2ZbEAPD2GW+y/EFX7HGhWW4iQeSW/wbG3hchJO6rShGRU912YZzot032TBhqvAKz7DWhRX4hG+ir9EZXPgkWRGdAo027BQ96gUbJsyY5Gk0oFe02anoQgHWRB/jX53QbQXpw0tsMdm06Mz9GsbBFVg0i0nRIPoYOyL5Dta5jyXRUeUeWLAp4xZjIhhTu/iAp6HneC3a2zelOIsnoj8PWzhgC2JgRPQ65nPP1Hk8Hs9/4hPZeVSuWENruAAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAXCAYAAABnGz2mAAABi0lEQVR4Xu2VuUrEUBSGjygW7oiFC7hgp1gJoo8gii9gY+mChfgADi6VaGUjNlqKiIULIohiJyKupXYqYmMl6DCo/+EehjuH3CTjZGSKfPDB5M+B/JN7kxDFOKmDlzosBNbhkw5dNOtAKIYJeAPP4R5stweypB8uUECxMtgHd8hc0IsleAur5HgCPsPa9ER4KuAi7CWfYmPwDe7DFHkXa4JJOGxlRfAVzlhZWBKwngKK2XySd7Fx+AO7VH4Gr63jSTjvcEpmeuCI/M652BqZYi0q51m+y6Uq92MWbpDZ+IfwQ37z8jpxFeNl5mINKt+W3PXABDFIOd6xEzIFeF/YbEreofIwDMEjMtdcgeWZpzNxFTsl/2KdKo8cVzHXUm5J3qryyOFiXEKzSqZAm8p59puy2/x/gosd6BCMkinWrXL+AtyrLC+4ijWSeS3YL1j+RL2TefzzSgn8gsf6hLAMr2C1HE/DF1iTnoiYAfhI5t/zcrH8qXmAldYc36E5eAcv4C79w6aPiYkpBH4BINNeS20b3zgAAAAASUVORK5CYII=>