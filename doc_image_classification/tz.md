## Название: Document Image Classification for Business Automation

## Задача: Классифицировать входящие сканы документов (счёт, договор, паспорт, накладная) — типичная задача в финтехе, логистике, госсекторе.  

## Датасет: RVL-CDIP
 (400k+ документов, 16 классов) или его подмножество.  

## Что сделать:

    Загрузить предобученную CNN (EfficientNet-B0 из timm).
    Fine-tune под 4 ключевых класса (например: invoice, contract, passport, waybill).
    Написать predict(image_path) → class + confidence.
    Обернуть в FastAPI с endpoint /predict.
    Добавить валидацию входа (размер, формат), обработку ошибок.
    Собрать Docker-образ, запустить локально.
    Написать README с примером curl-запроса.
