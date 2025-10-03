import os
import cv2
import numpy as np
import shutil
import base64
import torch
import xml.etree.ElementTree as ET
from xml.dom import minidom
import zipfile
import matplotlib.pyplot as plt
import io

class PanoramaProcessor:
    def __init__(self, panorama_filename, input_dir, output_dir, class_names):
        self.panorama_filename = panorama_filename
        self.panorama_base_name = os.path.splitext(panorama_filename)[0]
        self.input_dir = input_dir
        
        # --- Полные пути к папкам ---
        self.pano_output_dir = os.path.join(output_dir, self.panorama_base_name)
        self.discarded_masks_dir = os.path.join(self.pano_output_dir, "0_discarded_masks")
        self.source_masks_dir = os.path.join(self.pano_output_dir, "1_generated_masks")
        self.classified_masks_dir = os.path.join(self.pano_output_dir, "2_classified_masks")
        self.final_mask_dir = os.path.join(self.pano_output_dir, "3_final_mask")
        self.labelme_dir = os.path.join(self.pano_output_dir, "4_labelme_xml")
        self.final_zip_dir = os.path.join(self.pano_output_dir, "5_upload_to_cvat")
        
        self.class_names = class_names
        self.class_mapping = {name.strip().replace(" ", "_"): i + 1 for i, name in enumerate(class_names)}
        
        self.original_panorama_path = os.path.join(self.input_dir, self.panorama_filename)
        self.original_panorama = None
        self.mask_files = []
        self.status = "Готов к запуску"

        self.setup_and_load()

    def setup_and_load(self):
        """Создает необходимые папки и загружает панораму в память."""
        print(f"Настройка папок в: {self.pano_output_dir}")
        for path in [self.discarded_masks_dir, self.source_masks_dir, self.classified_masks_dir, 
                     self.final_mask_dir, self.labelme_dir, self.final_zip_dir]:
            os.makedirs(path, exist_ok=True)

        if not os.path.exists(self.original_panorama_path):
            raise FileNotFoundError(f"Панорама не найдена: {self.original_panorama_path}")
        
        self.original_panorama = cv2.imread(self.original_panorama_path)
        print("Панорама успешно загружена.")
        return True

    def _encode_image_to_base64(self, image_np):
        """Кодирует numpy-массив изображения в строку Base64 для передачи на фронтенд."""
        _, buffer = cv2.imencode('.jpg', image_np)
        return base64.b64encode(buffer).decode('utf-8')

    def generate_masks(self, sam_mask_generator, max_dimension=4000):
        """Генерирует маски, обновляя статус по ходу выполнения."""
        print("\n--- Шаг 1: Генерация масок ---")
        self.status = "Проверка существующих масок..."
        
        if os.listdir(self.source_masks_dir) or os.listdir(self.discarded_masks_dir):
            print("Маски уже сгенерированы. Пропускаем.")
            self.status = "Маски уже сгенерированы."
            self.mask_files = sorted([f for f in os.listdir(self.source_masks_dir) if f.endswith('.png')])
            return
        
        panorama_rgb = cv2.cvtColor(self.original_panorama, cv2.COLOR_BGR2RGB)
        full_height, full_width, _ = panorama_rgb.shape
        
        scale = max_dimension / max(full_height, full_width)
        target_width, target_height = int(full_width * scale), int(full_height * scale)
        
        print(f"Масштабирование до {target_width}x{target_height}...")
        self.status = f"Масштабирование изображения до {target_width}x{target_height}..."
        downscaled_panorama = cv2.resize(panorama_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        print("Запуск сегментации...")
        self.status = "Запуск сегментации (это самый долгий этап)..."
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        downscaled_masks = sam_mask_generator.generate(downscaled_panorama)
        print(f"Найдено {len(downscaled_masks)} 'сырых' масок.")

        print("Запуск пост-фильтрации...")
        self.status = f"Фильтрация {len(downscaled_masks)} найденных масок..."
        anns = sorted(downscaled_masks, key=lambda x: x['area'], reverse=True)
        filtered_anns_indices = []
        discarded_indices = set()
        for i in range(len(anns)):
            if i in discarded_indices: continue
            filtered_anns_indices.append(i)
            for j in range(i + 1, len(anns)):
                if j in discarded_indices: continue
                intersection = np.logical_and(anns[i]['segmentation'], anns[j]['segmentation']).sum()
                if intersection == 0: continue
                union = np.logical_or(anns[i]['segmentation'], anns[j]['segmentation']).sum()
                iou = intersection / union
                if iou > 0.9: 
                    discarded_indices.add(j)
                    continue
                containment_ratio = intersection / anns[j]['area']
                if containment_ratio > 0.95: 
                    discarded_indices.add(j)
        
        num_filtered = len(filtered_anns_indices)
        num_discarded = len(discarded_indices)
        print(f"Фильтрация завершена. Осталось {num_filtered} качественных масок, отброшено {num_discarded}.")
        
        self.status = f"Сохранение {num_discarded} отброшенных масок..."
        print(f"Сохранение {num_discarded} отфильтрованных масок...")
        for i, idx in enumerate(discarded_indices):
            mask_ann = anns[idx]
            small_mask = mask_ann['segmentation'].astype(np.uint8)
            full_size_mask = cv2.resize(small_mask, (full_width, full_height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.discarded_masks_dir, f"discarded_{i}.png"), full_size_mask * 255)

        self.status = f"Сохранение {num_filtered} качественных масок..."
        print(f"Сохранение {num_filtered} качественных масок для классификации...")
        for i, idx in enumerate(filtered_anns_indices):
            mask_ann = anns[idx]
            small_mask = mask_ann['segmentation'].astype(np.uint8)
            full_size_mask = cv2.resize(small_mask, (full_width, full_height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.source_masks_dir, f"mask_{i}.png"), full_size_mask * 255)
            
        self.mask_files = sorted([f for f in os.listdir(self.source_masks_dir) if f.endswith('.png')])
        print(f"Готово! Всего сохранено {len(self.mask_files)} масок для классификации.")
        self.status = "Генерация завершена."

    def get_mask_files_to_classify(self):
        """Возвращает АКТУАЛЬНЫЙ отсортированный список масок, ожидающих классификации."""
        self.mask_files = sorted([f for f in os.listdir(self.source_masks_dir) if f.endswith('.png')])
        return self.mask_files

    def get_mask_for_frontend(self, mask_filename):
        """Готовит данные для отображения одной маски на фронтенде."""
        filepath = os.path.join(self.source_masks_dir, mask_filename)
        if not os.path.exists(filepath):
            return None 

        mask_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        overlay = self.original_panorama.copy()
        highlight_color_bgr = (0, 255, 0)
        overlay[mask_image > 0] = highlight_color_bgr
        highlighted_image = cv2.addWeighted(self.original_panorama, 0.6, overlay, 0.4, 0)
        
        return {
            "mask_name": mask_filename,
            "highlighted_image_b64": self._encode_image_to_base64(highlighted_image),
        }

    def classify_mask(self, mask_filename, class_name):
        """Перемещает файл маски в папку соответствующего класса или удаляет его."""
        source_path = os.path.join(self.source_masks_dir, mask_filename)
        if not os.path.exists(source_path):
            return {"status": "error", "message": "Маска не найдена, возможно, уже обработана."}
            
        if class_name != "Пропустить":
            # Создаем папку для класса, если ее нет. Имя папки совпадает с текстом на кнопке.
            class_dir = os.path.join(self.classified_masks_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Перемещаем файл
            dest_path = os.path.join(class_dir, mask_filename)
            shutil.move(source_path, dest_path)
            print(f"Маска {mask_filename} классифицирована как '{class_name}'")
        else:
            # Просто удаляем файл, если он пропущен
            os.remove(source_path)
            print(f"Маска {mask_filename} пропущена")
            
        return {"status": "success"}

    def _build_final_mask(self):
        """Собирает единую маску из переименованных файлов."""
        print("Поиск классифицированных масок...")
        
        if not os.path.exists(self.classified_masks_dir):
             print("Папка с классифицированными масками не найдена.")
             return None

        # Просто получаем список всех .png файлов в папке
        classified_filenames = [f for f in os.listdir(self.classified_masks_dir) if f.endswith('.png')]
        
        if not classified_filenames:
            print("Не найдено ни одной классифицированной маски.")
            return None

        height, width, _ = self.original_panorama.shape
        final_mask = np.zeros((height, width), dtype=np.uint8)
        
        print(f"Собираем финальную маску из {len(classified_filenames)} файлов...")
        for filename in classified_filenames:
            try:
                # --- ИСПРАВЛЕНИЕ ЛОГИКИ ПАРСИНГА ---
                # Имя файла: 1_normals_Фон_0.png
                # split('_'): ['1', 'normals', 'Фон', '0.png']
                # Нам нужен третий элемент: 'Фон'
                class_name_clean = filename.split('_')[2]
                
                if class_name_clean in self.class_mapping:
                    class_id = self.class_mapping[class_name_clean]
                    filepath = os.path.join(self.classified_masks_dir, filename)
                    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        final_mask[mask > 0] = class_id
                else:
                    print(f"Класс '{class_name_clean}' из файла '{filename}' не найден в class_mapping.")

            except IndexError:
                print(f"Пропущен файл с некорректным именем (не удалось разделить на части): {filename}")
                continue
        
        return final_mask

    def visualize_final_mask(self):
        """Создает и возвращает визуализацию финальной маски."""
        final_mask = self._build_final_mask()
        
        if final_mask is None:
            return None

        print("Отрисовка результата в памяти...")
        fig, ax = plt.subplots(1, 1, figsize=(18, 9))
        im = ax.imshow(final_mask, cmap='tab20b')
        ax.set_title("Визуализация собранной маски")
        
        ticks = sorted(list(self.class_mapping.values()))
        tick_labels = [f"{i} - {name.replace('_', ' ')}" for name, i in sorted(self.class_mapping.items(), key=lambda item: item[1])]
        
        cbar = fig.colorbar(im, ax=ax, ticks=ticks)
        cbar.set_ticklabels(tick_labels)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        print("Визуализация готова для отправки на фронтенд.")
        return {"image_b64": img_b64}

    def create_final_dataset(self):
        """Создает ZIP-архив для CVAT."""
        final_mask = self._build_final_mask()

        if final_mask is None:
            raise ValueError("Нет классифицированных масок для создания датасета.")

        # Создание XML
        self._create_labelme_xml(final_mask)
        
        # Создание ZIP
        zip_filepath = os.path.join(self.final_zip_dir, f"{self.panorama_base_name}_for_cvat.zip")
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            zipf.write(self.original_panorama_path, arcname=self.panorama_filename)
            xml_filepath = os.path.join(self.labelme_dir, f"{self.panorama_base_name}.xml")
            if os.path.exists(xml_filepath):
                zipf.write(xml_filepath, arcname=f"{self.panorama_base_name}.xml")
        
        print(f"ZIP-архив готов для загрузки в CVAT: {zip_filepath}")
        return zip_filepath

    def _create_labelme_xml(self, final_mask):
        """Приватный метод для создания XML-файла."""
        height, width = final_mask.shape
        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'filename').text = self.panorama_filename
        ET.SubElement(annotation, 'folder').text = ''
        imagesize = ET.SubElement(annotation, 'imagesize')
        ET.SubElement(imagesize, 'nrows').text = str(height)
        ET.SubElement(imagesize, 'ncols').text = str(width)
        
        object_counter = 0
        for class_id in np.unique(final_mask):
            if class_id == 0: continue
            
            binary_mask = np.uint8(final_mask == class_id) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            try:
                # Находим имя класса по его ID
                class_name_with_underscore = list(self.class_mapping.keys())[list(self.class_mapping.values()).index(class_id)]
                original_class_name = class_name_with_underscore.replace("_", " ")
            except (ValueError, IndexError):
                continue
            
            for contour in contours:
                if cv2.contourArea(contour) < 5: continue
                obj = ET.SubElement(annotation, 'object')
                ET.SubElement(obj, 'name').text = original_class_name
                ET.SubElement(obj, 'deleted').text = '0'
                ET.SubElement(obj, 'id').text = str(object_counter)
                object_counter += 1
                polygon = ET.SubElement(obj, 'polygon')
                
                # Убедимся, что contour имеет правильную форму
                if contour.ndim == 3:
                    contour = contour.squeeze(axis=1)

                for point in contour:
                    pt = ET.SubElement(polygon, 'pt')
                    ET.SubElement(pt, 'x').text = str(int(point[0]))
                    ET.SubElement(pt, 'y').text = str(int(point[1]))

        xml_string = ET.tostring(annotation, 'utf-8')
        reparsed = minidom.parseString(xml_string)
        pretty_xml = '\n'.join(reparsed.toprettyxml(indent="  ").split('\n')[1:])
        
        xml_filepath = os.path.join(self.labelme_dir, f"{self.panorama_base_name}.xml")
        with open(xml_filepath, 'w', encoding='utf-8') as f: f.write(pretty_xml)
        print(f"XML-аннотация сохранена: {xml_filepath}")