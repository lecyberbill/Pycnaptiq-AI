from PIL import Image, ImageEnhance, ImageFilter, ImageChops
import numpy as np
from typing import Optional, Tuple


class ImageEditor:
    """
    Classe pour appliquer divers filtres et transformations à une image.
    """

    def __init__(self, image: Optional[np.ndarray]):
        """
        Initialise l'éditeur d'image avec une image.

        Args:
            image (np.ndarray): L'image d'entrée sous forme de tableau NumPy.
        """
        self.img = None
        self.load_image(image)

    def _check_image(self):
        """Vérifie si une image est chargée."""
        if self.img is None:
            raise ValueError("Aucune image chargée. Veuillez charger une image avec la méthode 'load_image'.")

    def _validate_image(self, image: np.ndarray):
        """
        Valide l'image d'entrée.

        Args:
            image (np.ndarray): L'image à valider.

        Raises:
            TypeError: Si l'image n'est pas un tableau NumPy.
            ValueError: Si l'image n'a pas les dimensions appropriées ou si le type de données est incorrect.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("L'image doit être un tableau NumPy.")

        if image.ndim not in (2, 3):
            raise ValueError("L'image doit être en 2D (niveaux de gris) ou 3D (couleur).")

        if image.dtype != np.uint8:
            raise ValueError("Le type de données de l'image doit être np.uint8.")

        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError("L'image couleur doit avoir 3 (RGB) ou 4 (RGBA) canaux.")

        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("La largeur et la hauteur de l'image doivent être positives.")

    def load_image(self, image: Optional[np.ndarray]):
        """Charge une nouvelle image dans l'éditeur.

        Args:
            image (np.ndarray): L'image à charger.
        """
        if image is None:
            self.img = None
        else:
            self._validate_image(image)
            self.img = Image.fromarray(image)

    def to_grayscale(self):
        """Convertit l'image en niveaux de gris."""
        self._check_image()
        self.img = self.img.convert("L")
        return self

    def rotate(self, angle: int):
        """Fait pivoter l'image d'un certain angle.

        Args:
            angle (int): L'angle de rotation en degrés.
        """
        self._check_image()
        if angle != 0:
            self.img = self.img.rotate(angle, expand=True)
        return self

    def mirror(self, mirror_type: str):
        """Applique un effet miroir à l'image.

        Args:
            mirror_type (str): Le type de miroir ("aucun", "horizontal", "vertical").
        """
        self._check_image()
        if mirror_type == "horizontal":
            self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror_type == "vertical":
            self.img = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        return self

    def blur(self, radius: int):
        """Applique un flou gaussien à l'image.

        Args:
            radius (int): Le rayon du flou.
        """
        self._check_image()
        if radius > 0:
            self.img = self.img.filter(ImageFilter.GaussianBlur(radius=radius))
        return self

    def sharpen(self, factor: float):
        """Applique un effet de netteté à l'image.

        Args:
            factor (float): Le facteur de netteté.
        """
        self._check_image()
        if factor > 0:
            enhancer = ImageEnhance.Sharpness(self.img)
            self.img = enhancer.enhance(factor)
        return self

    def adjust_contrast(self, factor: float):
        """Ajuste le contraste de l'image.

        Args:
            factor (float): Le facteur de contraste.
        """
        self._check_image()
        enhancer = ImageEnhance.Contrast(self.img)
        self.img = enhancer.enhance(factor)
        return self

    def adjust_saturation(self, factor: float):
        """Ajuste la saturation de l'image.

        Args:
            factor (float): Le facteur de saturation.
        """
        self._check_image()
        if self.img.mode != "L":  # Vérifie si l'image n'est pas en niveaux de gris
            enhancer = ImageEnhance.Color(self.img)
            self.img = enhancer.enhance(factor)
        return self

    def adjust_color_boost(self, factor: float):
        """Ajuste l'intensification des couleurs de l'image.

        Args:
            factor (float): Le facteur d'intensification des couleurs.
        """
        self._check_image()
        enhancer = ImageEnhance.Brightness(self.img)
        self.img = enhancer.enhance(factor)
        return self

    def apply_sepia(self):
        """Applique un filtre sépia à l'image."""
        self._check_image()
        if self.img.mode == "RGB":
            sepia_matrix = [
                0.393, 0.769, 0.189, 0,
                0.349, 0.686, 0.168, 0,
                0.272, 0.534, 0.131, 0
            ]
            self.img = self.img.convert("RGB", matrix=sepia_matrix)
        elif self.img.mode == "RGBA":
            sepia_matrix = [
                0.393, 0.769, 0.189, 0,
                0.349, 0.686, 0.168, 0,
                0.272, 0.534, 0.131, 0,
                0, 0, 0, 1
            ]
            self.img = self.img.convert("RGBA", matrix=sepia_matrix)
        return self

    def apply_contour(self):
        """Applique un filtre de contour à l'image."""
        self._check_image()
        self.img = self.img.filter(ImageFilter.CONTOUR)
        return self

    def apply_negative(self):
        """Applique un filtre négatif à l'image."""
        self._check_image()
        if self.img.mode == "L":
            self.img = Image.eval(self.img, lambda x: 255 - x)
        elif self.img.mode == "RGB":
            r, g, b = self.img.split()
            r = Image.eval(r, lambda x: 255 - x)
            g = Image.eval(g, lambda x: 255 - x)
            b = Image.eval(b, lambda x: 255 - x)
            self.img = Image.merge("RGB", (r, g, b))
        elif self.img.mode == "RGBA":
            r, g, b, a = self.img.split()
            r = Image.eval(r, lambda x: 255 - x)
            g = Image.eval(g, lambda x: 255 - x)
            b = Image.eval(b, lambda x: 255 - x)
            self.img = Image.merge("RGBA", (r, g, b, a))
        return self

    def apply_posterize(self, bits: int = 4):
        """Applique un filtre de postérisation à l'image.

        Args:
            bits (int): Le nombre de bits à utiliser pour la postérisation (1-8).
        """
        self._check_image()
        self.img = self.img.quantize(colors=2**bits)
        return self

    def apply_solarize(self, threshold: int = 128):
        """Applique un filtre de solarisation à l'image.

        Args:
            threshold (int): Le seuil de solarisation (0-255).
        """
        self._check_image()
        self.img = Image.eval(self.img, lambda x: 255 - x if x > threshold else x)
        return self

    def apply_emboss(self):
        """Applique un filtre de relief à l'image."""
        self._check_image()
        self.img = self.img.filter(ImageFilter.EMBOSS)
        return self

    def apply_pixelize(self, pixel_size: int = 8):
        """Applique un filtre de pixelisation à l'image.

        Args:
            pixel_size (int): La taille des pixels.
        """
        self._check_image()
        width, height = self.img.size
        self.img = self.img.resize((width // pixel_size, height // pixel_size), Image.NEAREST)
        self.img = self.img.resize((width, height), Image.NEAREST)
        return self

    def apply_vignette(self, radius: int = 100):
        """Applique un effet de vignettage à l'image.

        Args:
            radius (int): Le rayon du vignettage.
        """
        self._check_image()
        width, height = self.img.size
        mask = Image.new("L", (width, height), 0)
        for x in range(width):
            for y in range(height):
                dist_x = abs(x - width // 2)
                dist_y = abs(y - height // 2)
                dist = (dist_x**2 + dist_y**2)**0.5
                if dist > radius:
                    mask.putpixel((x, y), 255)
        self.img = Image.composite(Image.new("RGB", (width, height), "black"), self.img, mask)
        return self

    def apply_mosaic(self, block_size: int = 10):
        """Applique un effet de mosaïque à l'image.

        Args:
            block_size (int): La taille des blocs de la mosaïque.
        """
        self._check_image()
        width, height = self.img.size
        self.img = self.img.resize((width // block_size, height // block_size), Image.NEAREST)
        self.img = self.img.resize((width, height), Image.NEAREST)
        return self

########################################
###################    #FILTRE A TRAITER
########################################

    def adjust_vibrance(self, factor: float):
        """Ajuste la vibrance de l'image.

        Args:
            factor (float): Le facteur de vibrance.
        """
        self._check_image()
        if self.img.mode != "L":
            r, g, b = self.img.split()
            avg = ImageChops.add(r, ImageChops.add(g, b)).point(lambda x: x / 3)
            r = ImageChops.blend(r, avg, factor)
            g = ImageChops.blend(g, avg, factor)
            b = ImageChops.blend(b, avg, factor)
            self.img = Image.merge("RGB", (r, g, b))
        return self

    def adjust_hue(self, angle: int):
        """Ajuste la teinte de l'image.

        Args:
            angle (int): L'angle de modification de la teinte.
        """
        self._check_image()
        if self.img.mode != "L":
            self.img = self.img.convert("HSV")
            h, s, v = self.img.split()
            h = h.point(lambda x: (x + angle) % 256)
            self.img = Image.merge("HSV", (h, s, v)).convert("RGB")
        return self

    def adjust_curves(self, point1: float = 0.0, point2: float = 0.5, point3: float = 1.0):
        """Ajuste les courbes de l'image.

        Args:
            point1 (float): Valeur du point 1 (0.0 - 1.0).
            point2 (float): Valeur du point 2 (0.0 - 1.0).
            point3 (float): Valeur du point 3 (0.0 - 1.0).
        """
        self._check_image()
        # Créer la liste des points
        points = [(0.0, point1), (0.5, point2), (1.0, point3)]
        # Créer la LUT
        if self.img.mode != "L":
            lut = [0] * 256
            for i in range(256):
                # Normaliser la valeur d'entrée
                x = i / 255.0
                # Calculer la valeur de sortie en utilisant une interpolation linéaire
                if x < 0.5:
                    y = point1 + (point2 - point1) * (x / 0.5)                    
                else:
                    y = point2 + (point3 - point2) * ((x - 0.5) / 0.5)                    
                x = i / 255.0
                y = 0.0
                for p in points:
                    y += p[1] * (x ** p[0])
                lut[i] = int(max(0, min(255, y * 255)))
            if self.img.mode == "L":
                self.img = self.img.point(lut)
            elif self.img.mode == "RGB":
                r, g, b = self.img.split()
                r = r.point(lut)
                g = g.point(lut)
                b = b.point(lut)
                self.img = Image.merge("RGB", (r, g, b))
            elif self.img.mode == "RGBA":
                r, g, b, a = self.img.split()
                self.img = Image.merge("RGBA", (r.point(lut), g.point(lut), b.point(lut), a))
        return self

    def apply_unsharp_mask(self, radius: int = 0, percent: int = 100, threshold: int = 0):
        """Applique un filtre de netteté adaptative (Unsharp Mask) à l'image.

        Args:
            radius (int): Le rayon du flou.
            percent (int): Le pourcentage de netteté.
            threshold (int): Le seuil de différence pour appliquer la netteté.
        """
        self._check_image()
        if self.img.mode != "L":
            blurred = self.img.filter(ImageFilter.GaussianBlur(radius=radius))
            diff = ImageChops.difference(self.img, blurred)
            diff = diff.point(lambda x: 0 if x < threshold else x)
            self.img = ImageChops.add(self.img, diff.point(lambda x: x * percent / 100))
        return self

    def add_noise(self, amount: float = 0.0):
        """Ajoute du bruit (grain) à l'image.

        Args:
            amount (float): La quantité de bruit à ajouter (0.0 - 1.0).
        """
        self._check_image()
        if self.img.mode != "L":
            img_array = np.array(self.img)
            noise = np.random.normal(0, amount * 255, img_array.shape).astype(np.uint8)
            self.img = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
        return self

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convertit une couleur hexadécimale en tuple RGB."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def apply_color_gradient(self, color1: str = "#FF0000", color2: str = "#0000FF", gradient_active: bool = False, angle: int = 0):
        """Applique un dégradé de couleur à l'image.

        Args:
            color1 (tuple): La première couleur du dégradé (R, G, B).
            color2 (tuple): La deuxième couleur du dégradé (R, G, B).
        """
        self._check_image()
        if gradient_active:
            if self.img.mode != "L":
                rgb_color1 = self._hex_to_rgb(color1)
                rgb_color2 = self._hex_to_rgb(color2)
                width, height = self.img.size
                gradient = Image.new("RGB", (width, height))
                # Convertir l'angle en radians
                angle_rad = np.deg2rad(angle)
                
                # Calculer les composantes de la direction du dégradé
                dx = np.cos(angle_rad)
                dy = np.sin(angle_rad)
                
                # Calculer la distance maximale du dégradé
                max_dist = abs(width * dx) + abs(height * dy)
                
                for x in range(width):
                    for y in range(height):
                        dist = abs(x * dx + y * dy)
                        r = int(rgb_color1[0] + (rgb_color2[0] - rgb_color1[0]) * dist / max_dist)
                        g = int(rgb_color1[1] + (rgb_color2[1] - rgb_color1[1]) * dist / max_dist)
                        b = int(rgb_color1[2] + (rgb_color2[2] - rgb_color1[2]) * dist / max_dist)
                    for y in range(height):
                        gradient.putpixel((x, y), (r, g, b))
                self.img = Image.blend(self.img, gradient, 0.5)
        return self

    def apply_color_shift(self, r_shift: int = 0, g_shift: int = 0, b_shift: int = 0):
        """Décale les couleurs de l'image.

        Args:
            r_shift (int): Le décalage pour le canal rouge.
            g_shift (int): Le décalage pour le canal vert.
            b_shift (int): Le décalage pour le canal bleu.
        """
        self._check_image()
        if self.img.mode != "L":
            r, g, b = self.img.split()
            r = r.point(lambda x: (x + r_shift) % 256)
            g = g.point(lambda x: (x + g_shift) % 256)
            b = b.point(lambda x: (x + b_shift) % 256)
            self.img = Image.merge("RGB", (r, g, b))
        return self

    def apply_all_filters(self, contrast: float = 1.0, saturation: float = 1.0, color_boost: float = 1.0, grayscale: bool = False, blur_radius: int = 0, sharpness_factor: float = 1.0, rotation_angle: int = 0, mirror_type: str = "aucun", special_filter: str = "aucun", vibrance: float = 0.0, hue_angle: int = 0, point1: float = 0.0, point2: float = 0.5, point3: float = 1.0, unsharp_radius: int = 0, unsharp_percent: int = 100, unsharp_threshold: int = 0, noise_amount: float = 0.0, gradient_color1: str = "#FF0000", gradient_color2: str = "#0000FF", gradient_active: bool = False, color_shift_r: int = 0, color_shift_g: int = 0, color_shift_b: int = 0, gradient_angle: int = 0):
        """Applique tous les filtres en une seule fois.

        Args:
            contrast (float): Le facteur de contraste.
            saturation (float): Le facteur de saturation.
            color_boost (float): Le facteur d'intensification des couleurs.
            grayscale (bool): Si True, convertit l'image en niveaux de gris.
            blur_radius (int): Le rayon du flou.
            sharpness_factor (float): Le facteur de netteté.
            rotation_angle (int): L'angle de rotation.
            mirror_type (str): Le type de miroir.
            special_filter (str): Le filtre spécial à appliquer ("aucun", "sepia", "contour", "negative", "posterize", "solarize", "emboss", "pixelize", "vignette", "mosaic").
            vibrance (float): Le facteur de vibrance.
            hue_angle (int): L'angle de modification de la teinte.
            curves_points (list): Liste de points pour la courbe ([(x1, y1), (x2, y2), ...]).
            point1 (float): Valeur du point 1 (0.0 - 1.0).
            point2 (float): Valeur du point 2 (0.0 - 1.0).
            point3 (float): Valeur du point 3 (0.0 - 1.0).
            unsharp_radius (int): Le rayon du flou pour le filtre de netteté adaptative.
            unsharp_percent (int): Le pourcentage de netteté pour le filtre de netteté adaptative.
            unsharp_threshold (int): Le seuil de différence pour appliquer la netteté adaptative.
            noise_amount (float): La quantité de bruit à ajouter (0.0 - 1.0).
            gradient_color1 (tuple): La première couleur du dégradé (R, G, B).
            gradient_color2 (tuple): La deuxième couleur du dégradé (R, G, B).
            gradient_active (bool): Si True, applique le dégradé de couleur.
            gradient_angle (int): L'angle du dégradé.
            color_shift_r (int): Le décalage pour le canal rouge.
            color_shift_g (int): Le décalage pour le canal vert.
            color_shift_b (int): Le décalage pour le canal bleu.
        """
        if grayscale:
            self.to_grayscale()
        self.rotate(rotation_angle)
        self.mirror(mirror_type)
        self.blur(blur_radius)
        self.sharpen(sharpness_factor)
        self.adjust_contrast(contrast)
        self.adjust_saturation(saturation)
        self.adjust_color_boost(color_boost)

        # Appliquer le filtre spécial
        if special_filter == "sepia":
            self.apply_sepia()
        elif special_filter == "contour":
            self.apply_contour()
        elif special_filter == "negative":
            self.apply_negative()
        elif special_filter == "posterize":
            self.apply_posterize()
        elif special_filter == "solarize":
            self.apply_solarize()
        elif special_filter == "emboss":
            self.apply_emboss()
        elif special_filter == "pixelize":
            self.apply_pixelize()
        elif special_filter == "vignette":
            self.apply_vignette()
        elif special_filter == "mosaic":
            self.apply_mosaic()

        # Appliquer les nouveaux filtres
        self.adjust_vibrance(vibrance)
        self.adjust_hue(hue_angle)
        if point1 or point2 or point3:
            self.adjust_curves(point1, point2, point3)
        self.apply_unsharp_mask(radius=unsharp_radius, percent=unsharp_percent, threshold=unsharp_threshold)
        self.add_noise(amount=noise_amount)
        self.apply_color_gradient(color1=gradient_color1, color2=gradient_color2, gradient_active=gradient_active, angle=gradient_angle)
        self.apply_color_shift(r_shift=color_shift_r, g_shift=color_shift_g, b_shift=color_shift_b)

        return self


    def get_image(self) -> Optional[np.ndarray]:
        """Retourne l'image modifiée sous forme de tableau NumPy.

        Returns:
            np.ndarray: L'image modifiée.
        """
        if self.img is None:
            return None
        return np.array(self.img)
