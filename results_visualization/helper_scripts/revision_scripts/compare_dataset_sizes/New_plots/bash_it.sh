pdfcrop 'compare-dataset-sizes_Select-A_mem_footprint_x-mem_range_y-gflops.pdf'
pdfcrop 'compare-dataset-sizes_Select-avg_nz_row_x-anr_categ_y-gflops.pdf'
pdfcrop 'compare-dataset-sizes_Select-regularity_x-regularity_y-gflops.pdf'
pdfcrop 'compare-dataset-sizes_Select-skew_coeff_x-skew_categ_y-gflops.pdf'
pdftk 'compare-dataset-sizes_Select-A_mem_footprint_x-mem_range_y-gflops-crop.pdf' 'compare-dataset-sizes_Select-avg_nz_row_x-anr_categ_y-gflops-crop.pdf' 'compare-dataset-sizes_Select-skew_coeff_x-skew_categ_y-gflops-crop.pdf' 'compare-dataset-sizes_Select-regularity_x-regularity_y-gflops-crop.pdf' cat output 'Feature Impact.pdf'
pdfjam --suffix nup --nup 4x1 'Feature Impact.pdf'
pdfcrop 'Feature Impact-nup.pdf'
pdfcrop 'Feature Impact-nup-crop.pdf'
rm 'Feature Impact.pdf'
rm 'Feature Impact-nup.pdf'
rm 'Feature Impact-nup-crop.pdf'
mv 'Feature Impact-nup-crop-crop.pdf' 'Feature Impact.pdf'
# rm 'compare-dataset-sizes_Select-A_mem_footprint_x-mem_range_y-gflops-crop.pdf'
# rm 'compare-dataset-sizes_Select-avg_nz_row_x-anr_categ_y-gflops-crop.pdf'
# rm 'compare-dataset-sizes_Select-regularity_x-regularity_y-gflops-crop.pdf'
# rm 'compare-dataset-sizes_Select-skew_coeff_x-skew_categ_y-gflops-crop.pdf'
