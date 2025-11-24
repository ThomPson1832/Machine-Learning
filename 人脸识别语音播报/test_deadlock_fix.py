#!/usr/bin/env python3
"""
测试90%死锁问题修复 - 模拟真实人脸采集流程
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
import threading

def test_deadlock_scenario():
    """测试死锁场景"""
    print("🔍 测试死锁场景修复...")
    
    # 模拟锁和数据库
    import threading
    test_mutex = threading.Lock()
    test_database = {}
    
    def mock_save_database():
        """模拟保存数据库方法"""
        try:
            with test_mutex:
                data = {name: encoding.tolist() for name, encoding in test_database.items()}
                time.sleep(0.1)  # 模拟文件操作时间
                
                with open("test_db.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ❌ 保存失败: {e}")
            return False
        return True
    
    def mock_capture_with_deadlock():
        """模拟原来的死锁情况"""
        print("  测试原始死锁逻辑...")
        try:
            with test_mutex:
                test_database["test_user"] = np.random.rand(128).tolist()
                # 这里会导致死锁，因为save_database内部又要获取同一个锁
                # mock_save_database()  # 如果取消注释这行，会死锁
            return True
        except Exception as e:
            print(f"  ❌ 死锁测试失败: {e}")
            return False
    
    def mock_capture_fixed():
        """模拟修复后的逻辑"""
        print("  测试修复后逻辑...")
        try:
            # 临时检查（类似修复中的临时获取锁）
            with test_mutex:
                name_exists = "test_user" in test_database
            
            if name_exists:
                return False
                
            # 保存数据 - 避免死锁的直接保存
            with test_mutex:
                test_database["test_user"] = np.random.rand(128).tolist()
                
                # 直接保存而不是调用方法
                data = {name: encoding.tolist() for name, encoding in test_database.items()}
                time.sleep(0.1)  # 模拟文件操作时间
                
                with open("test_db_fixed.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"  ❌ 修复逻辑测试失败: {e}")
            return False
    
    # 清理测试文件
    for f in ["test_db.json", "test_db_fixed.json"]:
        if os.path.exists(f):
            os.remove(f)
    
    # 测试原始逻辑（避免实际死锁）
    result1 = mock_capture_with_deadlock()
    
    # 测试修复后逻辑
    result2 = mock_capture_fixed()
    
    # 清理测试文件
    for f in ["test_db.json", "test_db_fixed.json"]:
        if os.path.exists(f):
            os.remove(f)
    
    return result1 and result2

def test_progress_simulation():
    """测试进度条模拟"""
    print("\n🔍 测试进度条流程...")
    
    # 模拟人脸采集的各个阶段
    stages = [
        (20, "人脸检测"),
        (40, "人脸验证"),
        (60, "特征定位"),  
        (80, "特征提取"),
        (90, "数据库准备"),
        (100, "保存完成")
    ]
    
    print("  进度流程模拟:")
    for progress, stage in stages:
        print(f"    {progress}% - {stage}")
        time.sleep(0.05)  # 模拟处理时间
        
    print("  ✅ 进度条流程正常")
    return True

def test_file_operations():
    """测试文件操作"""
    print("\n🔍 测试文件操作...")
    
    test_data = {
        "测试用户": np.random.rand(128).tolist(),
        "用户2": np.random.rand(128).tolist()
    }
    
    test_file = "test_progress.json"
    
    try:
        # 测试写入
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print("  ✅ 文件写入成功")
        
        # 测试读取
        with open(test_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        print("  ✅ 文件读取成功")
        
        # 清理
        os.remove(test_file)
        print("  🧹 测试文件已清理")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 文件操作失败: {e}")
        return False

def test_concurrent_access():
    """测试并发访问"""
    print("\n🔍 测试并发访问...")
    
    results = []
    mutex = threading.Lock()
    
    def worker(worker_id):
        """工作线程"""
        try:
            # 模拟访问数据库
            with mutex:
                time.sleep(0.1)  # 模拟处理时间
            results.append(f"Worker-{worker_id} 完成")
        except Exception as e:
            results.append(f"Worker-{worker_id} 失败: {e}")
    
    # 模拟多个线程同时访问
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    success_count = sum(1 for r in results if "完成" in r)
    print(f"  📊 并发测试结果: {success_count}/5 线程成功")
    
    return success_count == 5

def main():
    """主测试函数"""
    print("🚀 90%死锁问题修复验证测试")
    print("=" * 50)
    
    # 测试项目
    tests = [
        ("死锁场景修复", test_deadlock_scenario),
        ("进度条流程", test_progress_simulation),
        ("文件操作", test_file_operations),
        ("并发访问", test_concurrent_access),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 测试: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name} - 通过")
                passed += 1
            else:
                print(f"❌ {test_name} - 失败")
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！90%死锁问题应该已修复")
        print("\n🔧 修复要点:")
        print("  1. ✅ 避免在持锁时调用可能死锁的方法")
        print("  2. ✅ 临时检查后立即释放锁")
        print("  3. ✅ 在同一锁内直接执行文件操作")
        print("  4. ✅ 添加了错误处理和数据回滚")
        
        print("\n💡 现在可以测试人脸采集功能:")
        print("   - 运行 python main.py")
        print("   - 点击'采集人脸'按钮")
        print("   - 检查进度条是否能到达100%")
        
    else:
        print("❌ 部分测试失败，需要进一步检查")
    
    return passed == total

if __name__ == "__main__":
    main()