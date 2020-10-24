#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>
#include <Eigen/LU>

using namespace Ubpa;

namespace {
	void ForwardPropagation(CanvasData* data, Eigen::MatrixXf &X) {
		Eigen::MatrixXf broadcastedB(data->opt_hidden_node_count, X.cols());
		broadcastedB.colwise() = data->b.col(0);
		data->Z = data->W.transpose() * X + broadcastedB; // z = w * x + b
		data->A = (data->Z.cwiseProduct(data->Z) * -0.5f).array().exp(); // a = g0(z)
		data->Y_predict = data->W2.transpose() * data->A + data->b2 * Eigen::MatrixXf::Ones(1, X.cols()); // y_predict = (w2 * a + b2)
	}

	float ComputeCost(CanvasData* data) {
		Eigen::MatrixXf T1, T2;
		switch (data->opt_loss_function) {
		case LossFunction::MeanSquaredError:
			T1 = data->Y - data->Y_predict; // t1 = y - y_predict
			return T1.cwiseProduct(T1).sum() / (2 * data->current_count); // cost = 1/2n * sum(t^2)
		case LossFunction::CrossEntropy:
			T1 = data->Y_predict.array().log(); // t1 = log(y_predict)
			T2 = (data->ONES - data->Y_predict).array().log(); // t2 = log(1 - y_predict)
			return (T1.cwiseProduct(data->Y) + T2.cwiseProduct(data->ONES - data->Y)).sum() / -(float)data->current_count; // cost = t1 * y + t2 * (1 - y)
		}
		return 0.f;
	}

	void BackwardPropagation(CanvasData* data) {
		Eigen::MatrixXf dY_predict;
		switch (data->opt_loss_function) {
		case LossFunction::MeanSquaredError:
			dY_predict = data->Y_predict - data->Y;
			break;
		case LossFunction::CrossEntropy:
			dY_predict = (data->ONES - data->Y).cwiseQuotient(data->ONES - data->Y_predict) - data->Y.cwiseQuotient(data->Y_predict);
			break;
		}
		data->dW2 = data->A * dY_predict.transpose();
		data->db2 = dY_predict.sum();
		Eigen::MatrixXf dZ = (data->W2 * dY_predict).cwiseProduct(data->A).cwiseProduct(data->Z * -1.f);
		data->dW = (dZ * data->X.transpose()).transpose();
		data->db = dZ.rowwise().sum();
	}
}

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<CanvasData>();
		if (!data)
			return;

		if (ImGui::Begin("Canvas")) {
			ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
			ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
			ImGui::Text("Mouse Left: click to add points,\nMouse Right: drag to scroll, click for context menu.");

			ImGui::SliderInt("Fitting Step", &data->opt_fitting_step, 1, 50);
			ImGui::SliderInt("Hidden Node Count", &data->opt_hidden_node_count, 5, 50);
			ImGui::SliderFloat("Learning Rate", &data->opt_learning_rate, 0.0001f, 1.f, "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);

			ImGui::Text("Loss Function: ");
			ImGui::SameLine();
			ImGui::RadioButton("Mean Squared Error", &data->opt_loss_function, LossFunction::MeanSquaredError);
			ImGui::SameLine();
			ImGui::RadioButton("Cross Entropy", &data->opt_loss_function, LossFunction::CrossEntropy);

			if (ImGui::Button(data->opt_is_training ? "Stop Training" : "Start Training")) data->opt_is_training = !data->opt_is_training;
			if (data->current_count) {
				ImGui::SameLine();
				if (ImGui::Button("Clear Model")) data->current_count = 0u;
				ImGui::SameLine();
				ImGui::Text("Current Input: %d, Cost: %.6f, Iteration: %d", data->current_count, data->cost, data->iteration);
			}

			// Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
			ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
			ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
			if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
			if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
			ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

			// Draw border and background color
			ImGuiIO& io = ImGui::GetIO();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
			draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

			// This will catch our interactions
			ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
			const bool is_hovered = ImGui::IsItemHovered(); // Hovered
			const bool is_active = ImGui::IsItemActive();   // Held
			const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin

			// use normalized position to reduce numerical errors
			const pointf2 mouse_pos_in_canvas((io.MousePos.x - origin.x) / canvas_sz.x, (io.MousePos.y - origin.y) / canvas_sz.y);

			// Add input point
			if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
				data->input_points.push_back(mouse_pos_in_canvas);
			}

			// Clear output lists
			data->point_set.clear();

			const size_t count = data->input_points.size();

			if (count && data->opt_is_training) {

				if (data->current_count != count || data->W.cols() != data->opt_hidden_node_count) { // re-init data if needed
					data->ONES = Eigen::MatrixXf::Ones(1, count);

					data->X.resize(1, count);
					data->Y.resize(1, count);
					for (size_t i = 0; i < count; i++) {
						data->X(0, i) = data->input_points[i][0];
						data->Y(0, i) = data->input_points[i][1];
					}

					// some reasonable initial values
					data->W = Eigen::MatrixXf::Random(1, data->opt_hidden_node_count).cwiseAbs() * 100.f;
					data->b = data->W.transpose().cwiseProduct(Eigen::MatrixXf::Random(data->opt_hidden_node_count, 1).cwiseAbs() * -1.f);
					data->W2 = Eigen::MatrixXf::Random(data->opt_hidden_node_count, 1).cwiseAbs() * 0.3f;
					data->b2 = 0.f;

					data->dW = Eigen::MatrixXf::Zero(1, data->opt_hidden_node_count);
					data->db = Eigen::MatrixXf::Zero(data->opt_hidden_node_count, 1);
					data->dW2 = Eigen::MatrixXf::Zero(data->opt_hidden_node_count, 1);
					data->db2 = 0.f;

					data->Z.resize(data->opt_hidden_node_count, count);
					data->A.resize(data->opt_hidden_node_count, count);
					data->Y_predict.resize(1, count);

					data->cost = 0.f;
					data->iteration = 0u;
					data->current_count = count;
				}

				ForwardPropagation(data, data->X);

				data->cost = ComputeCost(data);

				BackwardPropagation(data);

				// gradient descent
				data->W = data->W - data->opt_learning_rate * data->dW;
				data->b = data->b - data->opt_learning_rate * data->db;
				data->W2 = data->W2 - data->opt_learning_rate * data->dW2;
				data->b2 = data->b2 - data->opt_learning_rate * data->db2;

				data->iteration++;
			}

			// prediction
			if (data->current_count) {
				/* training data only *
				for (size_t i = 0; i < data->current_count; i++) {
					data->point_set.push_back(pointf2(data->X(0, i), data->Y_predict(0, i)));
				}
				/* full prediction set */
				float step = data->opt_fitting_step / canvas_sz.x;
				size_t targetCount = size_t(1.f / step);
				Eigen::MatrixXf X(1, targetCount);
				float currentX = 0.f;
				for (size_t i = 0u; i < targetCount; i++) {
					X(0, i) = currentX;
					currentX += step;
				}
				ForwardPropagation(data, X);
				for (size_t i = 0u; i < targetCount; i++) {
					data->point_set.push_back(pointf2(X(0, i), data->Y_predict(0, i)));
				}
				/* */
			}

			// Pan (we use a zero mouse threshold when there's no context menu)
			// You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
			const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
			if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan)) {
				data->scrolling[0] += io.MouseDelta.x;
				data->scrolling[1] += io.MouseDelta.y;
			}

			// Context menu (under default mouse threshold)
			ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
			if (data->opt_enable_context_menu && ImGui::IsMouseReleased(ImGuiMouseButton_Right) && drag_delta.x == 0.0f && drag_delta.y == 0.0f)
				ImGui::OpenPopupContextItem("context");
			if (ImGui::BeginPopup("context")) {
				if (ImGui::MenuItem("Remove one", NULL, false, data->input_points.size() > 0)) { data->input_points.resize(data->input_points.size() - 1); }
				if (ImGui::MenuItem("Remove all", NULL, false, data->input_points.size() > 0)) { data->input_points.clear(); }
				ImGui::EndPopup();
			}

			// Draw grid + all lines in the canvas
			draw_list->PushClipRect(canvas_p0, canvas_p1, true);
			if (data->opt_enable_grid) {
				const float GRID_STEP = 64.0f;
				for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
				for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
					draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
			}

			for (size_t n = 0; data->point_set.size() && n < data->point_set.size() - 1; n++) {
				pointf2& p1 = data->point_set[n], p2 = data->point_set[n + 1];
				draw_list->AddLine(
					ImVec2(origin.x + p1[0] * canvas_sz.x, origin.y + p1[1] * canvas_sz.y),
					ImVec2(origin.x + p2[0] * canvas_sz.x, origin.y + p2[1] * canvas_sz.y),
					IM_COL32(255, 0, 0, 255), 2.0f);
			}

			for (size_t n = 0; n < data->input_points.size(); n++) {
				pointf2& p = data->input_points[n];
				draw_list->AddCircleFilled(
					ImVec2(origin.x + p[0] * canvas_sz.x, origin.y + p[1] * canvas_sz.y),
					3.0f, IM_COL32(255, 255, 255, 255));
			}

			draw_list->PopClipRect();
		}

		ImGui::End();
	});
}
