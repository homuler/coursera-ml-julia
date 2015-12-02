using Images, ImageView

@doc """
  DISPLAYDATA Display 2D data in a nice grid
    [canvas, img] = DISPLAYDATA(X, example_width) displays 2D data
    stored in X in a nice grid. It returns the canvas and the
    displayed array.
""" ->
function displayData(X, example_width = round(Int, sqrt(size(X, 2))))
  # Compute rows, cols
  m, n = size(X)
  example_height = round(Int, (n / example_width))

  # Compute number of items to display
  display_rows = round(Int, sqrt(m))
  display_cols = round(Int, ceil(m / display_rows))

  # Between images padding
  pad = 1

  # Setup blank display
  display_array = - ones(pad + display_rows * (example_height + pad),
                         pad + display_cols * (example_width + pad))

  # Copy each example into a patch on the display array
  curr_ex = 1
  for j in 1:display_rows, i in 1:display_cols
		if curr_ex > m
			break
		end

		# Get the max value of the patch
		max_val = maximum(abs(X[curr_ex, :]))
		display_array[pad + (j - 1) * (example_height + pad) + (1:example_height),
		              pad + (i - 1) * (example_width + pad) + (1:example_width)] =
						reshape(X[curr_ex, :], (example_height, example_width)) / max_val
		curr_ex += 1
  end

  # Display Image
  img = Image(display_array)
  canvas = view(img)
  return (canvas, img)
end
