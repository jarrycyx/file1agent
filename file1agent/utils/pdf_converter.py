"""
PDF conversion utilities for converting PDF files to images.
"""

import os
import base64
import math
import traceback
from loguru import logger
import fitz


class PDFConverter:
    """
    Utility class for converting PDF files to images.
    """
    
    @staticmethod
    def convert_pdf_to_merged_image(pdf_file: str):
        """
        Convert all pages of a PDF file into a single merged image in a grid layout.
        
        Args:
            pdf_file: Path to the PDF file
            
        Returns:
            Tuple of (save_tmp_path, img_base64) or (None, None) if error
        """
        try:
            pdf_document = fitz.open(pdf_file)
            
            # Merge all pages into a single image
            # Calculate dimensions for the merged image in a grid layout
            page_images = []
            page_widths = []
            page_heights = []
            
            # First pass: render all pages and collect dimensions
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                page_images.append(pix)
                page_widths.append(pix.width)
                page_heights.append(pix.height)
            
            # Calculate grid layout to make the image more square
            num_pages = len(page_images)
            
            # Calculate average page dimensions
            avg_page_width = sum(page_widths) / len(page_widths)
            avg_page_height = sum(page_heights) / len(page_heights)
            aspect_ratio = avg_page_width / avg_page_height
            
            # Calculate optimal grid dimensions to minimize aspect ratio difference
            best_cols = int(math.ceil(math.sqrt(num_pages * aspect_ratio)))
            best_rows = int(math.ceil(num_pages / best_cols))
            
            # Try a few variations to find the most square layout
            min_diff = float('inf')
            for cols in range(max(1, best_cols - 2), best_cols + 3):
                rows = int(math.ceil(num_pages / cols))
                total_width = cols * max(page_widths)
                total_height = rows * max(page_heights)
                diff = abs(total_width - total_height) / max(total_width, total_height)
                if diff < min_diff:
                    min_diff = diff
                    best_cols = cols
                    best_rows = rows
            
            cols = best_cols
            rows = best_rows
            
            # Calculate max width and height for uniform grid cells
            max_page_width = max(page_widths)
            max_page_height = max(page_heights)
            
            # Calculate total dimensions for the merged image
            total_width = cols * max_page_width
            total_height = rows * max_page_height
            
            # Apply maximum resolution limit (4000 pixels)
            max_resolution = 4000
            if total_width > max_resolution or total_height > max_resolution:
                # Calculate scaling factor to fit within max_resolution
                scale_factor = min(max_resolution / total_width, max_resolution / total_height)
                
                # Scale down the total dimensions
                total_width = int(total_width * scale_factor)
                total_height = int(total_height * scale_factor)
                
                # Scale down the grid cell dimensions
                max_page_width = int(max_page_width * scale_factor)
                max_page_height = int(max_page_height * scale_factor)
                
                # Scale down each page image
                for i in range(len(page_images)):
                    # Get the original page and render it with scaling
                    page = pdf_document[i]
                    scale_matrix = fitz.Matrix(2 * scale_factor, 2 * scale_factor)  # 2x zoom for quality, then scale
                    scaled_pix = page.get_pixmap(matrix=scale_matrix)
                    page_images[i] = scaled_pix
            
            # Create a new pixmap for the merged image
            # First create a base pixmap from the first page
            first_pix = page_images[0]
            merged_pix = fitz.Pixmap(first_pix, total_width, total_height)
            
            # Create a single temporary document and page for all images
            temp_doc = fitz.open()
            temp_page = temp_doc.new_page(width=total_width, height=total_height)
            
            # Insert each page image to the temporary page in grid layout
            for i, pix in enumerate(page_images):
                # Calculate grid position
                row = i // cols
                col = i % cols
                
                # Calculate position in the grid
                x_offset = col * max_page_width
                y_offset = row * max_page_height
                
                # Center the page within its grid cell if needed
                x_center_offset = (max_page_width - pix.width) // 2
                y_center_offset = (max_page_height - pix.height) // 2
                
                # Insert the image at the calculated position
                temp_page.insert_image(
                    fitz.Rect(
                        x_offset + x_center_offset, 
                        y_offset + y_center_offset, 
                        x_offset + x_center_offset + pix.width, 
                        y_offset + y_center_offset + pix.height
                    ), 
                    pixmap=pix
                )
                pix = None  # Free memory
            
            # Get the final merged pixmap from the temporary page
            merged_pix = temp_page.get_pixmap()
            temp_doc.close()
            
            # Convert merged image to base64
            img_data = merged_pix.tobytes("png")
            img_base64 = base64.b64encode(img_data).decode("utf-8")
            
            try:
                if "workspace" in pdf_file:
                    save_path = pdf_file.split("workspace")[0]
                    save_tmp_path = os.path.join(save_path, "backup", "tmp", f"{os.path.basename(pdf_file)}_merged.png")
                else:
                    save_tmp_path = os.path.join(os.path.dirname(pdf_file), f".tmp_{os.path.basename(pdf_file)}_merged.png")
                os.makedirs(os.path.dirname(save_tmp_path), exist_ok=True)
                with open(save_tmp_path, "wb") as f:
                    f.write(base64.b64decode(img_base64))
            except Exception as e:
                error_msg = f"Error saving merged PDF image: {e}"
                logger.warning(error_msg)
                logger.error(traceback.format_exc())
            
            merged_pix = None  # Free memory
            pdf_document.close()
            return save_tmp_path, img_base64
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_file} to merged PNG: {e}")
            return None, None
    
    @staticmethod
    def convert_pdf_to_separate_images(pdf_file: str):
        """
        Convert each page of a PDF file into separate PNG images.
        
        Args:
            pdf_file: Path to the PDF file
            
        Returns:
            Tuple of (save_tmp_path, img_base64) for the first page or (None, None) if error
        """
        try:
            pdf_document = fitz.open(pdf_file)
            name_list, base64_list = [], []
            
            # Convert each page to PNG separately
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data).decode("utf-8")
                # Append page number to filename for identification
                
                try:
                    if "workspace" in pdf_file:
                        save_path = pdf_file.split("workspace")[0]
                        save_tmp_path = os.path.join(save_path, "backup", "tmp", f"{os.path.basename(pdf_file)}_page_{page_num+1}.png")
                    else:
                        save_tmp_path = os.path.join(os.path.dirname(pdf_file), f".tmp_{os.path.basename(pdf_file)}_page_{page_num+1}.png")
                    os.makedirs(os.path.dirname(save_tmp_path), exist_ok=True)
                    with open(save_tmp_path, "wb") as f:
                        f.write(base64.b64decode(img_base64))
                except Exception as e:
                    error_msg = f"Error saving separate PDF page image: {e}"
                    logger.warning(error_msg)
                    logger.error(traceback.format_exc())
                
                pdf_document.close()
                name_list.append(save_tmp_path)
                base64_list.append(img_base64)
            
            return name_list, base64_list
            
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_file} to separate PNGs: {e}")
            return name_list, base64_list
    
    @staticmethod
    def get_fig_base64(fig_file_list, merge_pdf=False):
        """
        Convert a list of figure files (images and PDFs) to base64 format.
        
        Args:
            fig_file_list: List of figure file paths
            merge_pdf: Whether to merge PDF pages into a single image
            
        Returns:
            List of tuples (file_path, base64_data)
        """
        fig_base64_list = []
        for fig in fig_file_list:
            # If it's a PDF file, convert it to PNG first
            if fig.lower().endswith(".pdf"):
                if merge_pdf:
                    name, img_base64 = PDFConverter.convert_pdf_to_merged_image(fig)
                    if name is None:
                        continue
                    fig_base64_list.append((name, img_base64))
                else:
                    name_list, img_base64_list = PDFConverter.convert_pdf_to_separate_images(fig)
                    if name_list is None:
                        continue
                    fig_base64_list.extend(zip(name_list, img_base64_list))
            else:
                with open(fig, "rb") as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode("utf-8")
                    fig_base64_list.append((fig, img_base64))
        return fig_base64_list
