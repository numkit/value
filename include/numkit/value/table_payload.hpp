// include/numkit/value/table_payload.hpp
//
// L0 opaque payload for tabular containers (`table` and `timetable`).
// Lives in numkit::value so both L2 runtime container implementations and
// L3 toolbox plugins (e.g. stats::fitlm / stats::predict) can inspect
// tabular values without any upward dependency on L2.
#pragma once

#include <numkit/value/error.hpp>
#include <numkit/value/object.hpp>
#include <numkit/value/value.hpp>

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/smart_ptr.h>
#include <keel/string.h>

namespace numkit {

struct TablePayload : NativePayload
{
    keel::usize nrows = 0;
    keel::Array<keel::String> varNames;
    keel::Array<Value> cols;
    keel::Array<keel::String> rowNames;
    keel::Array<keel::String> dimNames;
    keel::String description;
    keel::Array<keel::String> varUnits;
    keel::Array<keel::String> varDescriptions;
    bool isTimetable = false;
    Value rowTimes;
    keel::String rowTimesName;

    explicit TablePayload(keel::Allocator *alloc = keel::default_allocator())
        : varNames(alloc)
        , cols(alloc)
        , rowNames(alloc)
        , dimNames(alloc)
        , description(alloc)
        , varUnits(alloc)
        , varDescriptions(alloc)
        , rowTimesName(keel::String::from("Time", alloc))
    {
        dimNames.push_back(keel::String::from("Row", alloc));
        dimNames.push_back(keel::String::from("Variables", alloc));
    }

    keel::Arc<TablePayload> cloneTable(keel::Allocator *alloc = nullptr) const
    {
        auto *res = alloc ? alloc : keel::default_allocator();
        auto copy = keel::make_arc_in<TablePayload>(res, res);
        copy->nrows = nrows;
        copy->varNames = varNames.clone(res);
        copy->cols = cols.clone(res);
        copy->rowNames = rowNames.clone(res);
        copy->dimNames = dimNames.clone(res);
        copy->description = description.clone(res);
        copy->varUnits = varUnits.clone(res);
        copy->varDescriptions = varDescriptions.clone(res);
        copy->isTimetable = isTimetable;
        copy->rowTimes = rowTimes;
        copy->rowTimesName = rowTimesName.clone(res);
        return copy;
    }

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        return cloneTable(alloc);
    }

    keel::usize width() const { return cols.size(); }

    keel::i32 findVar(keel::StringRef name) const
    {
        for (keel::usize i = 0; i < varNames.size(); ++i)
            if (varNames[i] == name)
                return keel::as<keel::i32>(i);
        return -1;
    }

    keel::i32 findRow(keel::StringRef name) const
    {
        for (keel::usize i = 0; i < rowNames.size(); ++i)
            if (rowNames[i] == name)
                return keel::as<keel::i32>(i);
        return -1;
    }

    static keel::Array<keel::String> extractNames(const Value &v, const char *fn, const char *argName,
                                                  keel::Allocator *alloc = keel::default_allocator())
    {
        keel::Array<keel::String> out(alloc);
        if (v.isEmpty() && !v.isChar() && !v.isString())
            return out;
        if (v.isChar()) {
            if (v.dims().rows() <= 1) {
                if (v.numel() > 0)
                    out.push_back(keel::String::from(v.toString().as_ref(), alloc));
            } else {
                const char *cd = v.charData();
                keel::usize r = v.dims().rows(), c = v.dims().cols();
                for (keel::usize i = 0; i < r; ++i) {
                    keel::String s(alloc);
                    for (keel::usize j = 0; j < c; ++j)
                        s.push_back(cd[i + j * r]);
                    while (!s.empty() && s[s.size() - 1] == ' ')
                        s.pop_back();
                    out.push_back(keel::move(s));
                }
            }
            return out;
        }
        if (v.isString()) {
            for (keel::usize i = 0; i < v.numel(); ++i)
                out.push_back(keel::String::from(v.stringElem(i).as_ref(), alloc));
            return out;
        }
        if (v.isCell()) {
            for (keel::usize i = 0; i < v.numel(); ++i) {
                const Value &el = v.cellAt(i);
                if (!el.isChar() && !el.isString())
                    throw Error(keel::format("{}: '{}' must contain text strings", fn, argName),
                                0, 0, fn, "", keel::format("numkit:{}:invalidNames", fn));
                out.push_back(keel::String::from(el.toString().as_ref(), alloc));
            }
            return out;
        }
        throw Error(keel::format("{}: '{}' must be a cell array of char vectors or string array", fn, argName),
                    0, 0, fn, "", keel::format("numkit:{}:invalidNames", fn));
    }

    static void validateUniqueNames(const keel::Array<keel::String> &names,
                                    keel::usize expectedCount,
                                    const char *fn,
                                    const char *kind)
    {
        if (names.size() != expectedCount) {
            throw Error(keel::format("{}: number of {} ({}) must match expected count ({})",
                                     fn, kind, names.size(), expectedCount),
                        0, 0, fn, "", keel::format("numkit:{}:nameCountMismatch", fn));
        }
        for (keel::usize i = 0; i < names.size(); ++i) {
            if (names[i].empty())
                throw Error(keel::format("{}: {} cannot be empty", fn, kind),
                            0, 0, fn, "", keel::format("numkit:{}:emptyName", fn));
            for (keel::usize j = i + 1; j < names.size(); ++j) {
                if (names[i] == names[j])
                    throw Error(keel::format("{}: duplicate name '{}' in {}", fn, names[i], kind),
                                0, 0, fn, "", keel::format("numkit:{}:duplicateName", fn));
            }
        }
    }

    Value buildProperties(keel::Allocator *mr) const
    {
        Value st = Value::structure(mr);
        Value vn = Value::cell(1, varNames.size(), mr);
        for (keel::usize i = 0; i < varNames.size(); ++i)
            vn.cellAt(i) = Value::fromString(varNames[i], mr);
        st.field("VariableNames") = keel::move(vn);

        if (rowNames.empty()) {
            st.field("RowNames") = Value::cell(0, 0, mr);
        } else {
            Value rn = Value::cell(rowNames.size(), 1, mr);
            for (keel::usize i = 0; i < rowNames.size(); ++i)
                rn.cellAt(i) = Value::fromString(rowNames[i], mr);
            st.field("RowNames") = keel::move(rn);
        }

        Value dn = Value::cell(1, 2, mr);
        dn.cellAt(0) = Value::fromString(dimNames.size() > 0 ? dimNames[0].as_ref() : keel::StringRef("Row"), mr);
        dn.cellAt(1) = Value::fromString(dimNames.size() > 1 ? dimNames[1].as_ref() : keel::StringRef("Variables"), mr);
        st.field("DimensionNames") = keel::move(dn);

        st.field("Description") = Value::fromString(description, mr);

        if (varUnits.empty()) {
            st.field("VariableUnits") = Value::cell(1, 0, mr);
        } else {
            Value vu = Value::cell(1, varUnits.size(), mr);
            for (keel::usize i = 0; i < varUnits.size(); ++i)
                vu.cellAt(i) = Value::fromString(varUnits[i], mr);
            st.field("VariableUnits") = keel::move(vu);
        }

        if (varDescriptions.empty()) {
            st.field("VariableDescriptions") = Value::cell(1, 0, mr);
        } else {
            Value vd = Value::cell(1, varDescriptions.size(), mr);
            for (keel::usize i = 0; i < varDescriptions.size(); ++i)
                vd.cellAt(i) = Value::fromString(varDescriptions[i], mr);
            st.field("VariableDescriptions") = keel::move(vd);
        }
        return st;
    }

    void applyProperties(const Value &st)
    {
        if (!st.isStruct())
            throw Error("table: Properties must be a scalar struct",
                        0, 0, "table", "", "numkit:table:invalidProperties");
        if (st.hasField("VariableNames")) {
            auto vn = extractNames(st.field("VariableNames"), "table", "VariableNames");
            validateUniqueNames(vn, cols.size(), "table", "VariableNames");
            varNames = keel::move(vn);
        }
        if (st.hasField("RowNames")) {
            const Value &rnv = st.field("RowNames");
            if (rnv.isEmpty()) {
                rowNames.clear();
            } else {
                auto rn = extractNames(rnv, "table", "RowNames");
                validateUniqueNames(rn, nrows, "table", "RowNames");
                rowNames = keel::move(rn);
            }
        }
        if (st.hasField("DimensionNames")) {
            auto dn = extractNames(st.field("DimensionNames"), "table", "DimensionNames");
            validateUniqueNames(dn, 2, "table", "DimensionNames");
            dimNames = keel::move(dn);
        }
        if (st.hasField("Description")) {
            description.assign(st.field("Description").toString().as_ref());
        }
        if (st.hasField("VariableUnits")) {
            const Value &vuv = st.field("VariableUnits");
            if (vuv.isEmpty()) {
                varUnits.clear();
            } else {
                auto vu = extractNames(vuv, "table", "VariableUnits");
                if (vu.size() != cols.size())
                    throw Error("table: VariableUnits length must match table width",
                                0, 0, "table", "", "numkit:table:invalidVariableUnits");
                varUnits = keel::move(vu);
            }
        }
        if (st.hasField("VariableDescriptions")) {
            const Value &vdv = st.field("VariableDescriptions");
            if (vdv.isEmpty()) {
                varDescriptions.clear();
            } else {
                auto vd = extractNames(vdv, "table", "VariableDescriptions");
                if (vd.size() != cols.size())
                    throw Error("table: VariableDescriptions length must match table width",
                                0, 0, "table", "", "numkit:table:invalidVariableDescriptions");
                varDescriptions = keel::move(vd);
            }
        }
    }
};

inline const TablePayload *tablePayloadConst(const Value &v)
{
    if (!v.isObject() || (v.objectClassNameRef() != "table" && v.objectClassNameRef() != "timetable"))
        return nullptr;
    const auto *st = v.objectStateConst();
    return st ? static_cast<const TablePayload *>(st->native.get()) : nullptr;
}

inline TablePayload *tablePayloadMut(Value &v)
{
    if (!v.isObject() || (v.objectClassNameRef() != "table" && v.objectClassNameRef() != "timetable"))
        return nullptr;
    auto *st = v.objectStateMut();
    return st ? static_cast<TablePayload *>(st->native.get()) : nullptr;
}

inline Value makeTableValue(keel::Arc<TablePayload> payload, keel::Allocator *mr)
{
    if (!mr)
        mr = keel::default_allocator();
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    const char *clsName = (payload && payload->isTimetable) ? "timetable" : "table";
    st->native = keel::move(payload);
    return Value::object(clsName, keel::move(st), false, mr);
}

} // namespace numkit
