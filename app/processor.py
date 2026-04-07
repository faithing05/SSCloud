import os
import cv2
import numpy as np
import shutil
import base64
import json
import torch
import xml.etree.ElementTree as ET
from xml.dom import minidom
import zipfile
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import io
from datetime import datetime

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
        self.skipped_masks_dir = os.path.join(self.pano_output_dir, "2_skipped_masks")
        self.final_mask_dir = os.path.join(self.pano_output_dir, "3_final_mask")
        self.labelme_dir = os.path.join(self.pano_output_dir, "4_labelme_xml")
        self.final_zip_dir = os.path.join(self.pano_output_dir, "5_upload_to_cvat")
        self.manifest_path = os.path.join(self.pano_output_dir, "classification_manifest.json")
        
        self.class_names = class_names
        self.class_mapping = {name.strip().replace(" ", "_"): i + 1 for i, name in enumerate(class_names)}
        
        self.original_panorama_path = os.path.join(self.input_dir, self.panorama_filename)
        self.original_panorama = None
        self.mask_files = []
        self.status = "Готов к запуску"
        self.classification_manifest = {
            "panorama_filename": self.panorama_filename,
            "items": {},
            "action_history": [],
        }

        self.setup_and_load()

    def setup_and_load(self):
        """Создает необходимые папки и загружает панораму в память."""
        print(f"Настройка папок в: {self.pano_output_dir}")
        for path in [self.discarded_masks_dir, self.source_masks_dir, self.classified_masks_dir,
                     self.skipped_masks_dir,
                     self.final_mask_dir, self.labelme_dir, self.final_zip_dir]:
            os.makedirs(path, exist_ok=True)

        if not os.path.exists(self.original_panorama_path):
            raise FileNotFoundError(f"Панорама не найдена: {self.original_panorama_path}")
        
        self.original_panorama = cv2.imread(self.original_panorama_path)
        print("Панорама успешно загружена.")
        self._load_or_rebuild_manifest()
        return True

    def _timestamp(self):
        return datetime.utcnow().isoformat() + "Z"

    def _save_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(self.classification_manifest, manifest_file, ensure_ascii=False, indent=2)

    def _relative_path(self, absolute_path):
        return os.path.relpath(absolute_path, self.pano_output_dir).replace("\\", "/")

    def _absolute_path(self, relative_path):
        return os.path.join(self.pano_output_dir, relative_path.replace("/", os.sep))

    def _discover_classified_masks(self):
        discovered = {}
        if not os.path.exists(self.classified_masks_dir):
            return discovered

        for root, _, filenames in os.walk(self.classified_masks_dir):
            for filename in filenames:
                if not filename.endswith(".png"):
                    continue

                full_path = os.path.join(root, filename)
                rel_path = self._relative_path(full_path)
                rel_to_classified = os.path.relpath(full_path, self.classified_masks_dir)
                parts = rel_to_classified.split(os.sep)

                if len(parts) > 1:
                    class_name = parts[0]
                else:
                    class_name = None
                    name_without_ext = os.path.splitext(filename)[0]
                    name_parts = name_without_ext.split("_")
                    if len(name_parts) >= 3:
                        candidate = name_parts[2].replace("_", " ")
                        if candidate in self.class_names:
                            class_name = candidate

                if class_name:
                    discovered[filename] = {
                        "status": "classified",
                        "current_class": class_name,
                        "path": rel_path,
                    }

        return discovered

    def _load_or_rebuild_manifest(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as manifest_file:
                    data = json.load(manifest_file)
                    if isinstance(data, dict):
                        self.classification_manifest = {
                            "panorama_filename": data.get("panorama_filename", self.panorama_filename),
                            "items": data.get("items", {}),
                            "action_history": data.get("action_history", []),
                        }
            except Exception:
                self.classification_manifest = {
                    "panorama_filename": self.panorama_filename,
                    "items": {},
                    "action_history": [],
                }

        items = self.classification_manifest.get("items", {})

        pending_masks = [f for f in os.listdir(self.source_masks_dir) if f.endswith(".png")]
        for mask_name in pending_masks:
            mask_path = os.path.join(self.source_masks_dir, mask_name)
            current = items.get(mask_name, {})
            current.update({
                "status": "pending",
                "current_class": None,
                "path": self._relative_path(mask_path),
                "updated_at": self._timestamp(),
            })
            items[mask_name] = current

        skipped_masks = [f for f in os.listdir(self.skipped_masks_dir) if f.endswith(".png")]
        for mask_name in skipped_masks:
            mask_path = os.path.join(self.skipped_masks_dir, mask_name)
            current = items.get(mask_name, {})
            current.update({
                "status": "skipped",
                "current_class": None,
                "path": self._relative_path(mask_path),
                "updated_at": self._timestamp(),
            })
            items[mask_name] = current

        for mask_name, info in self._discover_classified_masks().items():
            current = items.get(mask_name, {})
            current.update({
                "status": "classified",
                "current_class": info["current_class"],
                "path": info["path"],
                "updated_at": self._timestamp(),
            })
            items[mask_name] = current

        self.classification_manifest["items"] = items
        self._save_manifest()

    def _resolve_mask_location(self, mask_filename):
        item = self.classification_manifest.get("items", {}).get(mask_filename)
        if item and item.get("path"):
            candidate = self._absolute_path(item["path"])
            if os.path.exists(candidate):
                return candidate

        source_candidate = os.path.join(self.source_masks_dir, mask_filename)
        if os.path.exists(source_candidate):
            return source_candidate

        skipped_candidate = os.path.join(self.skipped_masks_dir, mask_filename)
        if os.path.exists(skipped_candidate):
            return skipped_candidate

        for root, _, filenames in os.walk(self.classified_masks_dir):
            if mask_filename in filenames:
                return os.path.join(root, mask_filename)

        return None

    def _encode_image_to_base64(self, image_np, image_format='.jpg'):
        """Кодирует numpy-массив изображения в строку Base64 для передачи на фронтенд."""
        _, buffer = cv2.imencode(image_format, image_np)
        return base64.b64encode(buffer).decode('utf-8')

    def generate_masks(self, sam_mask_generator, max_dimension=4000):
        """Генерирует маски, обновляя статус по ходу выполнения."""
        print("\n--- Шаг 1: Генерация масок ---")
        self.status = "Проверка существующих масок..."
        
        if os.listdir(self.source_masks_dir) or os.listdir(self.discarded_masks_dir):
            print("Маски уже сгенерированы. Пропускаем.")
            self.status = "Маски уже сгенерированы."
            self.mask_files = sorted([f for f in os.listdir(self.source_masks_dir) if f.endswith('.png')])
            self._load_or_rebuild_manifest()
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
        self._load_or_rebuild_manifest()
        print(f"Готово! Всего сохранено {len(self.mask_files)} масок для классификации.")
        self.status = "Генерация завершена."

    def get_mask_files_to_classify(self):
        """Возвращает АКТУАЛЬНЫЙ отсортированный список масок, ожидающих классификации."""
        self._load_or_rebuild_manifest()
        self.mask_files = sorted([f for f in os.listdir(self.source_masks_dir) if f.endswith('.png')])
        return self.mask_files

    def get_mask_for_frontend(self, mask_filename):
        """Готовит данные для отображения одной маски на фронтенде."""
        filepath = self._resolve_mask_location(mask_filename)
        if not filepath or not os.path.exists(filepath):
            return None 

        mask_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        overlay = self.original_panorama.copy()
        highlight_color_bgr = (0, 255, 0)
        overlay[mask_image > 0] = highlight_color_bgr
        highlighted_image = cv2.addWeighted(self.original_panorama, 0.6, overlay, 0.4, 0)
        
        return {
            "mask_name": mask_filename,
            "original_panorama_b64": self._encode_image_to_base64(self.original_panorama),
            "mask_image_b64": self._encode_image_to_base64(mask_image, image_format='.png'),
            "highlighted_image_b64": self._encode_image_to_base64(highlighted_image),
        }

    def get_review_items(self):
        self._load_or_rebuild_manifest()
        review_items = []
        for mask_name, item in self.classification_manifest.get("items", {}).items():
            status = item.get("status")
            if status not in {"classified", "skipped"}:
                continue

            review_items.append({
                "mask_name": mask_name,
                "status": status,
                "class_name": item.get("current_class"),
                "updated_at": item.get("updated_at"),
            })

        return sorted(review_items, key=lambda entry: entry["mask_name"])

    def get_classification_state(self):
        self._load_or_rebuild_manifest()
        items = self.classification_manifest.get("items", {})
        pending = 0
        classified = 0
        skipped = 0

        for item in items.values():
            status = item.get("status")
            if status == "pending":
                pending += 1
            elif status == "classified":
                classified += 1
            elif status == "skipped":
                skipped += 1

        return {
            "panorama": self.panorama_filename,
            "pending": pending,
            "classified": classified,
            "skipped": skipped,
            "history_depth": len(self.classification_manifest.get("action_history", [])),
        }

    def classify_mask(self, mask_filename, class_name, record_history=True):
        """Перемещает файл маски в папку соответствующего класса или удаляет его."""
        self._load_or_rebuild_manifest()
        source_path = self._resolve_mask_location(mask_filename)
        if not source_path or not os.path.exists(source_path):
            return {"status": "error", "message": "Маска не найдена, возможно, уже обработана."}

        item = self.classification_manifest.get("items", {}).get(mask_filename, {})
        previous_status = item.get("status", "pending")
        previous_class = item.get("current_class")
        previous_path = item.get("path", self._relative_path(source_path))

        if class_name not in self.class_names and class_name != "Пропустить":
            return {"status": "error", "message": "Неизвестный класс."}

        if class_name == "Пропустить":
            os.makedirs(self.skipped_masks_dir, exist_ok=True)
            dest_path = os.path.join(self.skipped_masks_dir, mask_filename)
            new_status = "skipped"
            new_class = None
        else:
            class_dir = os.path.join(self.classified_masks_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            dest_path = os.path.join(class_dir, mask_filename)
            new_status = "classified"
            new_class = class_name

        if os.path.abspath(source_path) != os.path.abspath(dest_path):
            shutil.move(source_path, dest_path)

        item.update({
            "status": new_status,
            "current_class": new_class,
            "path": self._relative_path(dest_path),
            "updated_at": self._timestamp(),
        })
        self.classification_manifest["items"][mask_filename] = item

        if record_history:
            self.classification_manifest["action_history"].append({
                "mask_name": mask_filename,
                "previous_status": previous_status,
                "previous_class": previous_class,
                "previous_path": previous_path,
                "new_status": new_status,
                "new_class": new_class,
                "new_path": self._relative_path(dest_path),
                "updated_at": self._timestamp(),
            })

        self._save_manifest()
        self.mask_files = sorted([f for f in os.listdir(self.source_masks_dir) if f.endswith('.png')])

        if class_name != "Пропустить":
            print(f"Маска {mask_filename} классифицирована как '{class_name}'")
        else:
            print(f"Маска {mask_filename} пропущена")
            
        return {"status": "success"}

    def undo_last_classification(self):
        self._load_or_rebuild_manifest()
        history = self.classification_manifest.get("action_history", [])
        if not history:
            return {"status": "error", "message": "Нет действий для отмены."}

        action = history.pop()
        mask_name = action.get("mask_name")
        previous_path = action.get("previous_path")

        current_path = self._resolve_mask_location(mask_name)
        if not current_path or not os.path.exists(current_path):
            return {"status": "error", "message": f"Не удалось найти маску для отмены: {mask_name}"}

        restore_path = self._absolute_path(previous_path)
        os.makedirs(os.path.dirname(restore_path), exist_ok=True)
        if os.path.abspath(current_path) != os.path.abspath(restore_path):
            shutil.move(current_path, restore_path)

        item = self.classification_manifest.get("items", {}).get(mask_name, {})
        item.update({
            "status": action.get("previous_status", "pending"),
            "current_class": action.get("previous_class"),
            "path": previous_path,
            "updated_at": self._timestamp(),
        })
        self.classification_manifest["items"][mask_name] = item
        self._save_manifest()

        return {
            "status": "success",
            "mask_name": mask_name,
            "restored_status": item.get("status"),
            "restored_class": item.get("current_class"),
        }

    def _build_final_mask(self):
        """Собирает единую маску из переименованных файлов."""
        print("Поиск классифицированных масок...")
        
        if not os.path.exists(self.classified_masks_dir):
             print("Папка с классифицированными масками не найдена.")
             return None

        classified_files = []
        for root, _, filenames in os.walk(self.classified_masks_dir):
            for filename in filenames:
                if filename.endswith('.png'):
                    class_name = os.path.basename(root)
                    classified_files.append((class_name, os.path.join(root, filename)))
        
        if not classified_files:
            print("Не найдено ни одной классифицированной маски.")
            return None

        height, width, _ = self.original_panorama.shape
        final_mask = np.zeros((height, width), dtype=np.uint8)
        
        print(f"Собираем финальную маску из {len(classified_files)} файлов...")
        for class_name, filepath in classified_files:
            try:
                if class_name in self.class_mapping:
                    class_id = self.class_mapping[class_name]
                    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        final_mask[mask > 0] = class_id
                else:
                    print(f"Класс '{class_name}' не найден в class_mapping.")

            except IndexError:
                print(f"Пропущен файл с некорректной структурой: {filepath}")
                continue
        
        return final_mask

    def visualize_final_mask(self):
        """Создает и возвращает визуализацию финальной маски."""
        final_mask = self._build_final_mask()
        
        if final_mask is None:
            return None

        print("Отрисовка результата в памяти...")
        max_class_id = max(self.class_mapping.values(), default=0)
        class_palette = [
            "#000000",  # 0 - фон
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffff33",
            "#a65628",
            "#f781bf",
        ]
        if max_class_id >= len(class_palette):
            extended_cmap = plt.get_cmap("tab20", max_class_id + 1)
            class_palette = [extended_cmap(i) for i in range(max_class_id + 1)]

        cmap = ListedColormap(class_palette[: max_class_id + 1])
        boundaries = np.arange(-0.5, max_class_id + 1.5, 1)
        norm = BoundaryNorm(boundaries, cmap.N)

        fig, ax = plt.subplots(1, 1, figsize=(18, 9))
        ax.imshow(final_mask, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title("Визуализация собранной маски")
        ax.axis("off")

        id_to_class = {class_id: class_name for class_name, class_id in self.class_mapping.items()}
        present_class_ids = [int(class_id) for class_id in np.unique(final_mask) if class_id != 0]
        present_class_ids.sort()

        legend_handles = []
        for class_id in present_class_ids:
            class_name = id_to_class.get(class_id, f"Класс {class_id}").replace("_", " ")
            legend_handles.append(
                Patch(
                    facecolor=class_palette[class_id],
                    edgecolor="black",
                    label=f"{class_id} - {class_name}",
                )
            )

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title="Классы",
                loc="upper right",
                fontsize=8,
                title_fontsize=9,
                framealpha=0.95,
                borderpad=0.3,
                labelspacing=0.3,
                handlelength=1.2,
                handleheight=0.8,
            )
        
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
