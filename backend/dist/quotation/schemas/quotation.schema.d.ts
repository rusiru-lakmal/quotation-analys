import { Document, Schema as MongooseSchema } from 'mongoose';
export declare class Quotation extends Document {
    originalFile: string;
    extractedData: any;
    status: string;
}
export declare const QuotationSchema: MongooseSchema<Quotation, import("mongoose").Model<Quotation, any, any, any, Document<unknown, any, Quotation, any, {}> & Quotation & Required<{
    _id: import("mongoose").Types.ObjectId;
}> & {
    __v: number;
}, any>, {}, {}, {}, {}, import("mongoose").DefaultSchemaOptions, Quotation, Document<unknown, {}, import("mongoose").FlatRecord<Quotation>, {}, import("mongoose").DefaultSchemaOptions> & import("mongoose").FlatRecord<Quotation> & Required<{
    _id: import("mongoose").Types.ObjectId;
}> & {
    __v: number;
}>;
