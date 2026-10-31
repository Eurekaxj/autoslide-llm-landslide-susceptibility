# -*- coding: utf-8 -*-
import arcpy
import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment


def setup_arcpy_environment(output_dir):
    """设置 arcpy 工作环境"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    arcpy.env.overwriteOutput = True
    arcpy.env.workspace = output_dir

    # 初始化 Spatial Analyst 许可（只执行一次）
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
        return True
    else:
        print("Spatial Analyst扩展不可用")
        return False


def create_random_points(run_num, params, log_callback):
    """
    在指定范围内创建随机点
    :param run_num: 当前轮次编号
    :param params: 参数字典 {'constraining_feature', 'output_dir', 'num_points', 'min_distance'}
    :param log_callback: 日志回调函数
    :return: 生成的点文件路径 or None
    """
    log_callback(u"[开始] 创建第{}批随机点...".format(run_num))

    constraining_feature = params.get('sample_area')
    num_points = params.get('sample_count')
    min_distance = "{} Meters".format(params.get('sample_distance'))

    output_points = os.path.join(params['output_dir'], "RandomPoints_{}.shp".format(run_num))

    try:
        arcpy.CreateRandomPoints_management(
            out_path=os.path.dirname(output_points),
            out_name=os.path.basename(output_points),
            constraining_feature_class=constraining_feature,
            number_of_points_or_field=num_points,
            minimum_allowed_distance=min_distance
        )
        log_callback(u"[完成] 第{}批随机点创建成功: {}".format(run_num, output_points))
        return output_points
    except arcpy.ExecuteError as e:
        error_msg = arcpy.GetMessages(2)
        log_callback(u"[失败] 第{}批随机点创建失败:\n{}".format(run_num, error_msg))
        return None


def extract_values_to_points(random_points, raster_paths, log_callback):
    """
    提取多个栅格图层的属性值到点
    :param random_points: 输入点文件路径
    :param raster_paths: 栅格文件路径列表
    :param log_callback: 日志回调函数
    :return: 成功与否
    """
    log_callback(u"[开始] 提取属性值到点: {}".format(random_points))

    input_rasters = [[raster, "NONE"] for raster in raster_paths]

    try:
        arcpy.gp.ExtractMultiValuesToPoints_sa(
            in_point_features=random_points,
            in_rasters=input_rasters
        )
        log_callback(u"[完成] 属性提取完成: {}".format(random_points))
        return True
    except arcpy.ExecuteError as e:
        error_msg = arcpy.GetMessages(2)
        log_callback(u"[失败] 提取属性值失败:\n{}".format(error_msg))
        return False


def calculate_lon_lat(point_feature, log_callback):
    """
    计算经纬度字段
    :param point_feature: 点要素类路径
    :param log_callback: 日志回调函数
    :return: 成功与否
    """
    log_callback(u"[开始] 开始计算经纬度")

    try:
        # 添加经纬度字段
        for field in ["Lon", "Lat"]:
            if field not in [f.name for f in arcpy.ListFields(point_feature)]:
                arcpy.AddField_management(point_feature, field, "DOUBLE")

        # 创建WGS84地理坐标系（用于经纬度计算）
        wgs84_sr = arcpy.SpatialReference(4326)

        # 计算经纬度
        with arcpy.da.UpdateCursor(point_feature, ["SHAPE@", "Lon", "Lat"]) as cursor:
            for row in cursor:
                point = row[0].projectAs(wgs84_sr)
                row[1] = point.firstPoint.X  # 经度
                row[2] = point.firstPoint.Y  # 纬度
                cursor.updateRow(row)

        log_callback(u"[完成] 经纬度计算完成")
        return True
    except Exception as e:
        log_callback(u"[失败] 计算经纬度时出错: {}".format(str(e)))
        return False


def export_to_excel(points_with_attributes, run_num, template_path, output_dir, log_callback):
    """
    导出点属性表到 Excel，并合并模板数据
    :param points_with_attributes: 点文件路径
    :param run_num: 轮次号
    :param template_path: 模板Excel路径
    :param output_dir: 输出目录
    :param log_callback: 日志回调函数
    :return: Excel输出路径 or None
    """
    log_callback(u"[开始] 导出属性表并生成合并后的新Excel...")

    temp_excel = os.path.join(os.path.dirname(points_with_attributes), "temp_{}.xls".format(run_num))
    try:
        arcpy.TableToExcel_conversion(points_with_attributes, temp_excel)
        log_callback(u"[完成] 临时Excel导出成功: {}".format(temp_excel))
    except arcpy.ExecuteError:
        error_msg = arcpy.GetMessages(2)
        log_callback(u"[失败] 导出Excel时出错:\n{}".format(error_msg))
        return None

    # 读取临时导出的Excel
    try:
        df_new = pd.read_excel(temp_excel)
    except Exception as e:
        log_callback(u"[失败] 读取临时Excel失败: {}".format(str(e)))
        return None

    # 读取模板原始landslidedata
    try:
        df_template = pd.read_excel(template_path)
    except Exception as e:
        log_callback(u"[失败] 读取原始landslidedata失败: {}".format(str(e)))
        return None

    # 补全列 & 添加特定字段
    template_columns = df_template.columns.tolist()
    for col in template_columns:
        if col not in df_new.columns:
            df_new[col] = 0

    df_new["landslide"] = 0
    df_new["possibility"] = 0

    if "id" in df_template.columns:
        max_id = df_template["id"].max()
    else:
        max_id = 0
    df_new["id"] = range(max_id + 1, max_id + 1 + len(df_new))
    df_new = df_new[template_columns]

    df_combined = pd.concat([df_template, df_new], ignore_index=True)

    # 输出路径
    output_excel = os.path.join(output_dir, "DataSet__{}.xlsx".format(run_num))

    try:
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            df_combined.to_excel(writer, index=False, sheet_name="Sheet1")

        wb = load_workbook(output_excel)
        ws = wb["Sheet1"]

        alignment = Alignment(horizontal="center", vertical="center")
        float_format = "0.0000"

        for col in ws.columns:
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
            adjusted_width = max_length + 2
            col_letter = col[0].column_letter
            ws.column_dimensions[col_letter].width = adjusted_width

            for cell in col:
                cell.alignment = alignment
                if isinstance(cell.value, float):
                    cell.number_format = float_format

        wb.save(output_excel)
        log_callback(u"[完成] 新数据集生成成功: {}".format(output_excel))
        return output_excel
    except Exception as e:
        log_callback(u"[失败] 保存或格式化Excel失败: {}".format(str(e)))
        return None